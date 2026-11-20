"""
v3 on top of results_centre/v2/ (simulation only), results in results_centre/v3/:

    python centre_v3.py prepare            # cache + labels of dataset_v3/ (40k train2 + 5k test2 presses)
    python centre_v3.py bank               # structured-noise reconstructions of the train2 presses (as v2)
    python centre_v3.py train --size 20000 [--seed 0]   # v2 recipe on 2k / 7.2k / 20k / 40k presses
    python centre_v3.py curve              # evaluate all sizes on the new 5k test set (bootstrap CIs) + plots
    python centre_v3.py seeds              # mean +- std over seeds (test metrics + search loop)
    python centre_v3.py abs-calibrate      # fit the CAD baseline's 4 constants on validation presses
    python centre_v3.py compare            # CAD baseline vs v2 on the same test presses and perturbation grid
    python centre_v3.py search ...         # search loop on test2 starts with a v2 model or the CAD baseline
    python centre_v3.py runtime            # time per press, CPU and GPU, network and baseline

Training sets are nested: 2k = the first 2,000 of the 7,217 natural training presses of dataset/
(7.2k = all of them = results_centre/v2/model.pt), 20k and 40k add the first 12,783 / 32,783
presses of dataset_v3/train2. Same augmentation, same 120 epochs (so the larger sets also get more
optimizer steps), same validation set (dataset/ val, natural) for picking the epoch.

GelSight Mini field of view (18.6 x 14.4 mm instead of 32 x 24 mm), the same commands with GELSIGHT_SENSOR=mini on
dataset_mini/ -> results_centre/mini/ (sizes 20k / 40k / 100k of its train2; validation = its own val split):

    python collect_dataset.py collect --sensor mini --out dataset_mini --only-split val --n 1000
    python collect_dataset.py collect --sensor mini --out dataset_mini --only-split train2 --n 100000
    python collect_dataset.py collect --sensor mini --out dataset_mini --only-split test2 --n 5000
    GELSIGHT_SENSOR=mini python centre_v3.py prepare | bank | train --size 100000 | curve | search ... | seeds
    python centre_v3.py fov-compare        # standard 40k (3 seeds) vs mini, each on its own 5k test set

Input resampling: augment_v2 resamples every training input (bilinear, zoom 0.97-1.03, never exactly 1); the
test inputs are not resampled, and every model is worse on them than on the same inputs zoomed by 0.99:

    python centre_v3.py input-scale --resample 0.99; GELSIGHT_SENSOR=mini python centre_v3.py input-scale --resample 0.99
    python centre_v3.py fov-compare --resample 0.99
    python centre_v3.py search --conditions resampled_0.99 --runs ...
"""
from __future__ import annotations

import argparse
import json
import pickle
import os
import sys
import time

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import numpy as np

import centre_study as cs
import centre_v2 as v2
import train_centre as tc

HERE = os.path.dirname(os.path.abspath(__file__))
D3 = os.path.join(HERE, "dataset_v3")
CACHE3 = os.path.join(D3, "cache_v3.npz")
BANK3 = os.path.join(D3, "cache_v3_bank.npy")
V3 = os.path.join(tc.OUT, "v3")
SIZES = (2000, 7217, 20000, 40000)
# GELSIGHT_SENSOR=mini: the same pipeline for the GelSight Mini field of view (18.6 x 14.4 mm) on dataset_mini/
# (val = 1k presses with dataset/ val's seeds, train2 = 100k, test2 = 5k with dataset_v3's seeds), results in
# results_centre/mini/. Nested training sets: the first N presses of train2; validation: dataset_mini val.
MINI = tc.SENSOR == "mini"
# CENTRE_AUG_EXACT=0.25: that fraction of the training inputs is not resampled (augment_v2 p_exact); 0 = the v2 recipe.
# Needs its own output folder (CENTRE_MINI_OUT), so the v2-recipe models are never overwritten.
AUG_EXACT = float(os.environ.get("CENTRE_AUG_EXACT", "0"))
# CENTRE_AUG_ROBUST=1: centre_v2.augment_robust (random sub-pixel shift, interpolation kernel and border treatment);
# the model then masks the outer TEST_BORDER_PX px of every input at test time (load_net). Own output folder too.
AUG_ROBUST = os.environ.get("CENTRE_AUG_ROBUST", "0") == "1"
# CENTRE_INPUT_BLUR_PX=s: a fixed Gaussian blur (sigma s input px) of the depth channel in front of the network, in
# training (after the augmentation and the border) and at test time (load_net, json input_blur_px). Own folder too.
INPUT_BLUR = float(os.environ.get("CENTRE_INPUT_BLUR_PX", "0"))
if MINI:   # CENTRE_MINI_DATA / CENTRE_MINI_OUT: other folders (smoke tests)
    D3 = os.environ.get("CENTRE_MINI_DATA", os.path.join(HERE, "dataset_mini"))
    CACHE3 = os.path.join(D3, "cache_mini.npz")
    BANK3 = os.path.join(D3, "cache_mini_bank.npy")
    V3 = os.environ.get("CENTRE_MINI_OUT", os.path.join(tc.OUT, "mini"))
    SIZES = (20000, 40000, 100000)


def model_path(size, seed=0):
    if MINI:
        return os.path.join(V3, f"model_{size // 1000}k_seed{seed}.pt")
    if size == 7217 and seed == 0:
        return os.path.join(v2.V2, "model.pt")          # the v2 model is the 7.2k point
    return os.path.join(V3, f"model_{size // 1000}k_seed{seed}.pt" if size != 7217 else f"model_7k_seed{seed}.pt")


# ================================================================ cache + bank
def prepare():
    import glob
    import h5py
    import cv2
    t0 = time.time()
    keys = ["seed", "sample_type", "pose_id", "pose_category", "would_move", "contact_fork", "contact_table", "com_s",
            "bbox_s", "fork_pos_s", "fork_quat_s", "n_tines", "force_cmd", "target_part"]
    cols = {k: [] for k in keys + ["X", "split", "index", "shard", "part"]}
    for p in sorted(glob.glob(os.path.join(D3, "*-[0-9][0-9][0-9][0-9].h5"))):
        with h5py.File(p) as f:
            n = int(f.attrs["n"])
            rec = f["depth_recon"][()].astype(np.float32)
            cols["X"].append(np.stack([cv2.resize(r, (tc.W_IN, tc.H_IN), interpolation=cv2.INTER_AREA) for r in rec])
                             .astype(np.float16))
            cols["split"].append(np.full(n, {"val": 1, "train2": 3, "test2": 4}[f.attrs["split"]]))
            cols["index"].append(np.arange(n))
            cols["shard"].append(np.full(n, os.path.basename(p)))
            for k in keys:
                cols[k].append(f[k][()])
            cols["part"].append(tc.dominant_parts(f))
        print(f"  {os.path.basename(p)} ({time.time() - t0:.0f} s)", flush=True)
    C = {k: np.concatenate(v) for k, v in cols.items()}
    np.savez(CACHE3, **C)
    print(f"wrote {CACHE3}: {len(C['seed'])} presses")


def load3():
    C = dict(np.load(CACHE3))
    C["cls"] = np.where(C["contact_fork"], 2, np.where(C["contact_table"], 1, 0))
    C["vec_mm"] = C["com_s"][:, :2] * 1000.0
    C["natural"] = C["pose_category"] == 0
    return C


def _bank_worker3(args):
    shard, idx, seed0 = args
    import h5py
    pm = v2._pm()
    with h5py.File(os.path.join(D3, shard)) as f:
        ref, rest = f["reference_image"][()], f["marker_rest"][()]
        out = np.zeros((len(idx), v2.K_VARIANTS, tc.H_IN, tc.W_IN), np.float16)
        for j, i in enumerate(idx):
            img = f["img"][i]
            mk = rest + f["marker_shift"][i]
            for k in range(v2.K_VARIANTS):
                rng = np.random.default_rng([seed0, int(i), k, 3])
                I2, R2 = v2.structured_noise(img, ref, rng, mk, **v2.random_variant(rng))
                out[j, k] = cs.downsample(pm.height_map(I2, R2, v2.MM_PER_PX_CAM)[0]).astype(np.float16)
    return shard, idx, out


def build_bank3(workers=24):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    C = load3()
    rows = np.nonzero(C["split"] == 3)[0]
    bank = np.lib.format.open_memmap(BANK3, mode="w+", dtype=np.float16,
                                     shape=(len(rows), v2.K_VARIANTS, tc.H_IN, tc.W_IN))
    pos = {(C["shard"][g], int(C["index"][g])): n for n, g in enumerate(rows)}
    tasks = []
    for shard in sorted(set(C["shard"][rows])):
        r = rows[C["shard"][rows] == shard]
        for part in np.array_split(r, 2):
            tasks.append((shard, C["index"][part].tolist(), 3030))
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for k, (shard, idx, out) in enumerate(ex.map(_bank_worker3, tasks)):
            for j, i in enumerate(idx):
                bank[pos[(shard, i)]] = out[j]
            if k % 10 == 0:
                print(f"  {k + 1}/{len(tasks)} chunks ({time.time() - t0:.0f} s)", flush=True)
    bank.flush()
    print(f"wrote {BANK3}: {bank.shape}, {time.time() - t0:.0f} s")


# ================================================================ training on a chosen size
def training_arrays(size):
    if MINI:
        return training_arrays_mini(size)
    C = tc.load_cache()
    tr_old = np.nonzero((C["split"] == 0) & C["natural"])[0]
    bank_old = np.load(v2.BANK, mmap_mode="r")
    rows_old = np.load(v2.BANK_ROWS)
    bpos = np.full(len(C["seed"]), -1)
    bpos[rows_old] = np.arange(len(rows_old))
    n_old = min(size, len(tr_old))
    sel_old = tr_old[:n_old]
    X = [C["X"][sel_old]]
    B = [np.asarray(bank_old[bpos[sel_old]])]
    lab = {k: [C[k][sel_old]] for k in ("cls", "part", "pose_id", "vec_mm")}
    if size > len(tr_old):
        C3 = load3()
        tr_new = np.nonzero(C3["split"] == 3)[0][:size - len(tr_old)]
        bank_new = np.load(BANK3, mmap_mode="r")
        X.append(C3["X"][tr_new])
        B.append(np.asarray(bank_new[tr_new - np.nonzero(C3["split"] == 3)[0][0]]))
        for k in lab:
            lab[k].append(C3[k][tr_new])
    va = np.nonzero((C["split"] == 1) & C["natural"])[0]
    val = dict(X=C["X"][va], **{k: C[k][va] for k in ("cls", "part", "pose_id", "vec_mm")})
    return np.concatenate(X), np.concatenate(B), {k: np.concatenate(v) for k, v in lab.items()}, val


def training_arrays_mini(size):
    """dataset_mini: the first `size` train2 presses (all natural, static fork) and their bank rows; val split."""
    C = load3()
    tr_all = np.nonzero(C["split"] == 3)[0]
    if size > len(tr_all):
        raise SystemExit(f"only {len(tr_all)} train2 presses in {D3}")
    tr = tr_all[:size]
    bank = np.load(BANK3, mmap_mode="r")
    if len(bank) != len(tr_all):
        raise SystemExit(f"{BANK3} has {len(bank)} rows, train2 has {len(tr_all)}: run `bank` again")
    lab = {k: C[k][tr] for k in ("cls", "part", "pose_id", "vec_mm")}
    va = np.nonzero((C["split"] == 1) & C["natural"])[0]
    val = dict(X=C["X"][va], **{k: C[k][va] for k in ("cls", "part", "pose_id", "vec_mm")})
    return C["X"][tr], np.asarray(bank[:size]), lab, val


def _labels(d, device):
    import torch
    pose = d["pose_id"]
    return dict(cls=torch.tensor(d["cls"], device=device),
                part=torch.tensor(np.where(d["part"] >= 0, d["part"], 0), device=device),
                pose=torch.tensor(np.clip(pose, 0, 1), device=device),
                pose_known=torch.tensor(pose <= 1, device=device),
                vec=torch.tensor(d["vec_mm"], dtype=torch.float32, device=device),
                fork=torch.tensor(d["cls"] == 2, device=device))


CKPT_EVERY = int(os.environ.get("CENTRE_CKPT_EVERY", "5"))


def train_size(size, seed=0, epochs=120, bs=64, lr=2e-3):
    import torch
    os.makedirs(V3, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    path = model_path(size, seed)
    if AUG_EXACT > 0 and os.path.realpath(V3) in {os.path.realpath(os.path.join(tc.OUT, d)) for d in ("v3", "mini")}:
        raise SystemExit("CENTRE_AUG_EXACT > 0 needs its own CENTRE_MINI_OUT folder (would overwrite v2-recipe models)")
    if AUG_ROBUST and (AUG_EXACT > 0 or os.path.realpath(V3) in {os.path.realpath(os.path.join(tc.OUT, d))
                                                                for d in ("v3", "mini", "mini_exact")}):
        raise SystemExit("CENTRE_AUG_ROBUST=1 needs its own CENTRE_MINI_OUT folder and no CENTRE_AUG_EXACT")
    if INPUT_BLUR and os.path.realpath(V3) in {os.path.realpath(os.path.join(tc.OUT, d))
                                               for d in ("v3", "mini", "mini_exact", "mini_robust", "mini_robust_tta")}:
        raise SystemExit("CENTRE_INPUT_BLUR_PX needs its own CENTRE_MINI_OUT folder")
    logf = open(os.path.join(V3, "train.log"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"     # cpu: smoke tests only
    Xc, Bk, lab, val = training_arrays(size)
    n = len(Xc)
    Ltr = _labels(lab, dev)
    Xva = torch.tensor(val["X"].astype(np.float32), device=dev)[:, None]
    if AUG_ROBUST:                                  # select the epoch on inputs as the model will see them
        Xva = v2.mask_border(Xva)
    Xva = v2.input_blur(Xva, INPUT_BLUR)
    Lva = _labels(val, dev)
    coords = tc.coord_channels(dev)
    model = tc.build_model().to(dev).to(memory_format=torch.channels_last)    # 1.5x faster on this GPU
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(n / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    gen = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed)
    ckpt_path, ep0 = path.replace(".pt", ".ckpt"), 0
    best, hist, t_prev = None, [], 0.0
    if os.path.exists(ckpt_path):                   # resume after a crash (checkpoint every CKPT_EVERY epochs)
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"].to(gen.get_state().device) if hasattr(ck["gen"], "to") else ck["gen"])
        rng.bit_generator.state = ck["rng"]
        ep0, best, hist, t_prev = ck["epoch"], ck["best"], ck["hist"], ck["seconds"]
    log(f"v3 train size {size} ({n} presses), seed {seed}, {epochs} epochs, un-resampled fraction {AUG_EXACT}"
        + (f", robust augmentation (test border: outer {v2.TEST_BORDER_PX} px masked)" if AUG_ROBUST else "")
        + (f", input blur sigma {INPUT_BLUR} px" if INPUT_BLUR else "")
        + f" -> {os.path.relpath(path, HERE)}" + (f" (resumed after epoch {ep0})" if ep0 else ""))
    t0 = time.time() - t_prev
    for ep in range(ep0, epochs):
        model.train()
        perm = rng.permutation(n)
        tot = {k: 0.0 for k in "abcd"}
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            vsel = rng.integers(0, v2.K_VARIANTS + 1, len(idx))
            xb = np.where((vsel == v2.K_VARIANTS)[:, None, None], Xc[idx], Bk[idx, np.minimum(vsel, v2.K_VARIANTS - 1)])
            x = torch.tensor(xb.astype(np.float32), device=dev)[:, None]
            it = torch.tensor(idx, device=dev)
            lb = {k: t[it] for k, t in Ltr.items()}
            if AUG_ROBUST:
                x, vr, _ = v2.augment_robust(x, lb["vec"], gen)
            else:
                x, vr = v2.augment_v2(x, lb["vec"], gen, p_exact=AUG_EXACT)
            x = v2.input_blur(x, INPUT_BLUR)
            x = torch.cat([x, coords.expand(len(x), -1, -1, -1)], 1).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                out = model(x)
            L = v2.losses_v2([o.float() for o in out], lb, vr)
            loss = L["a"] + L["b"] + L["c"] + L["d"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            for k in "abcd":
                tot[k] += L[k].item() * len(idx)
        vl = tc.evaluate_loss(model, Xva, {k: Lva[k] for k in ("cls", "part", "pose", "vec", "fork")}, coords)
        hist.append(dict(epoch=ep + 1, train={k: tot[k] / n for k in "abcd"}, val=vl))
        score = vl["a"] + vl["b"] + vl["c"] + vl["d"]
        if best is None or score < best[0]:
            best = (score, ep + 1)
            torch.save(model.state_dict(), path)
        if (ep + 1) % 5 == 0 or ep == 0:
            log(f"  [{size // 1000}k s{seed}] epoch {ep + 1:3d}: train d {tot['d'] / n:.3f} | val a {vl['a']:.3f} "
                f"b {vl['b']:.3f} c {vl['c']:.3f} d {vl['d']:.3f} ({time.time() - t0:.0f} s)")
        with open(path.replace(".pt", ".heartbeat"), "w") as f:     # for the watchdog: written every epoch
            f.write(f"{ep + 1} {time.time() - t0:.0f}\n")
        if (ep + 1) % CKPT_EVERY == 0 and ep + 1 < epochs:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), gen=gen.get_state(),
                            rng=rng.bit_generator.state, epoch=ep + 1, best=best, hist=hist, seconds=time.time() - t0),
                       ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
    log(f"  [{size // 1000}k s{seed}] best epoch {best[1]} (clean val loss {best[0]:.3f}), {time.time() - t0:.0f} s")
    with open(path.replace(".pt", ".json"), "w") as f:
        json.dump(dict(size=size, seed=seed, n=n, best_epoch=best[1], best_val_loss=best[0], epochs=epochs, resumed_from=ep0,
                       aug_exact=AUG_EXACT, aug_robust=AUG_ROBUST,
                       test_border_px=v2.TEST_BORDER_PX if AUG_ROBUST else 0, input_blur_px=INPUT_BLUR,
                       seconds=time.time() - t0, history=hist), f, indent=1)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


# ================================================================ evaluation on the new 5k test set
def load_net(path, device):
    """The network; for a model trained with augment_robust (its json says test_border_px > 0) wrapped so that
    every input gets the fixed test border treatment, whichever code path feeds it."""
    import torch
    m = tc.build_model().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    js = path.replace(".pt", ".json")
    meta = json.load(open(js)) if os.path.exists(js) else {}
    px = meta.get("test_border_px", 0)
    if meta.get("tta"):                             # test-time augmentation: inside the border mask and the blur
        m = tta_wrap(m, meta["tta"]).to(device).eval()
    if meta.get("input_blur_px"):                   # fixed input blur, after the border mask (as in training)
        class InputBlurred(torch.nn.Module):
            def __init__(self, net, sigma):
                super().__init__()
                self.net, self.sigma = net, float(sigma)

            def forward(self, x):
                return self.net(v2.input_blur(x, self.sigma))
        m = InputBlurred(m, meta["input_blur_px"]).to(device).eval()
    if px:
        class BorderMasked(torch.nn.Module):
            """Sets the outer px rows / columns of the depth channel to 0 before the network (the coordinate
            channels are left alone)."""

            def __init__(self, net, px):
                super().__init__()
                self.net, self.px = net, int(px)

            def forward(self, x):
                return self.net(v2.mask_border(x, self.px))
        m = BorderMasked(m, px).to(device).eval()
    return m


TTA_DEFAULT = [["bilinear", 0.5, 0.5, 1.0], ["bilinear", -0.5, -0.5, 1.0], ["bilinear", 0.5, -0.5, 1.0],
               ["bilinear", -0.5, 0.5, 1.0], ["bicubic", 0.5, 0.5, 1.0], ["bicubic", -0.5, -0.5, 1.0],
               ["bicubic", 0.5, -0.5, 1.0], ["bicubic", -0.5, 0.5, 1.0], ["bilinear", 0.0, 0.0, 0.99],
               ["bilinear", 0.0, 0.0, 1.01], ["identity", 0.0, 0.0, 1.0]]


def tta_wrap(net, variants):
    """Average the four heads' probabilities over resamplings of the depth channel: (kernel, shift x px, shift y px,
    zoom) about the pad centre, zeros outside (as augment_robust). Shifts come in opposite pairs, so the averaged
    heatmap is not displaced. Returns log-probabilities, so callers' softmax gives the average."""
    import torch
    import torch.nn.functional as F

    class TTA(torch.nn.Module):
        def __init__(self, net, variants):
            super().__init__()
            self.net, self.variants = net, [tuple(v) for v in variants]

        def forward(self, x):
            B, _, H, W = x.shape
            S = torch.tensor([W * tc.MM_PER_PX_IN / 2, H * tc.MM_PER_PX_IN / 2], device=x.device)
            acc = None
            for kern, dx, dy, z in self.variants:
                if kern == "identity":
                    xv = x
                else:
                    th = torch.zeros(1, 2, 3, device=x.device)
                    th[0, 0, 0], th[0, 1, 1] = 1 / z, 1 / z
                    th[0, 0, 2] = -dx * tc.MM_PER_PX_IN / z / S[0]
                    th[0, 1, 2] = -dy * tc.MM_PER_PX_IN / z / S[1]
                    g = F.affine_grid(th.expand(B, 2, 3), (B, 1, H, W), align_corners=False)
                    d = F.grid_sample(x[:, :1], g, mode=kern, padding_mode="zeros", align_corners=False)
                    xv = torch.cat([d, x[:, 1:]], 1)
                out = [torch.softmax(o.float(), 1) for o in self.net(xv)]
                acc = out if acc is None else [a + o for a, o in zip(acc, out)]
            return [torch.log(a / len(self.variants) + 1e-30) for a in acc]
    return TTA(net, variants)


def per_sample(model, C, idx, X=None):
    """Per-press outcomes on the selected test presses (fork rows carry the heatmap numbers)."""
    Cx = C if X is None else dict(C, X=_with(C, idx, X))
    pa, pb, pc, P = tc.predict(model, Cx, idx, bs=128)
    fork = C["cls"][idx] == 2
    hm = tc.heat_metrics(P[fork], C["vec_mm"][idx][fork])
    out = dict(idx=idx, a_ok=pa.argmax(1) == C["cls"][idx], fork=fork, pa_fork=pa[:, 2])
    fi = idx[fork]
    out.update(err=hm["argmax_err_mm"], hit=hm["cover_90"], area=hm["area_90_mm2"], nll=hm["nll"],
               cover50=hm["cover_50"], cover80=hm["cover_80"], cover95=hm["cover_95"],
               b_ok=pb[fork].argmax(1) == C["part"][fi], c_ok=pc[fork].argmax(1) == C["pose_id"][fi],
               part=C["part"][fi], pose=C["pose_id"][fi], would_move=C["would_move"][fi])
    return out


def _with(C, idx, X):
    Xa = C["X"].copy()
    Xa[idx] = np.asarray(X).astype(np.float16)
    return Xa


def summarise_ci(r, B=1000, seed=0):
    """Point estimate + 95 % bootstrap interval per group (all, parts, part x pose)."""
    rng = np.random.default_rng(seed)
    groups = {"all": np.ones(len(r["err"]), bool)}
    for k, p in enumerate(tc.PARTS4):
        groups[p] = r["part"] == k
        for j, q in enumerate(tc.POSES):
            groups[f"{p}, {q}"] = (r["part"] == k) & (r["pose"] == j)
    groups["raised tines (face up, would_move)"] = (r["part"] == 0) & (r["pose"] == 1) & r["would_move"]
    stats = dict(median_mm=lambda e, h, a, b, c: np.median(e), p90_mm=lambda e, h, a, b, c: np.percentile(e, 90),
                 hit90=lambda e, h, a, b, c: h.mean(), area90_mm2=lambda e, h, a, b, c: np.median(a),
                 part_acc=lambda e, h, a, b, c: b.mean(), pose_acc=lambda e, h, a, b, c: c.mean())
    out = {}
    for g, sel in groups.items():
        n = int(sel.sum())
        if n < 5:
            continue
        arr = [r[k][sel] for k in ("err", "hit", "area", "b_ok", "c_ok")]
        d = dict(n=n)
        boots = rng.integers(0, n, (B, n))
        for name, f in stats.items():
            est = float(f(*arr))
            bs = np.array([f(*[x[b] for x in arr]) for b in boots])
            d[name] = [est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out[g] = d
    a = r["a_ok"]
    bs = np.array([a[b].mean() for b in rng.integers(0, len(a), (B, len(a)))])
    out["contact_acc"] = [float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def curve(seeds=(0,)):
    import torch
    os.makedirs(V3, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C = load3()
    test = np.nonzero(C["split"] == 4)[0]
    res = {}
    for size in SIZES:
        for seed in seeds:
            path = model_path(size, seed)
            if not os.path.exists(path):
                print("missing", path)
                continue
            r = per_sample(load_net(path, dev), C, test)
            np.savez_compressed(os.path.join(V3, f"test2_{size // 1000}k_seed{seed}.npz"), **r)
            res[f"{size}/{seed}"] = summarise_ci(r)
            a = res[f"{size}/{seed}"]["all"]
            print(f"size {size:6d} seed {seed}: median {a['median_mm'][0]:.2f} [{a['median_mm'][1]:.2f}, {a['median_mm'][2]:.2f}] "
                  f"p90 {a['p90_mm'][0]:.2f} hit {a['hit90'][0]:.3f} area {a['area90_mm2'][0]:.0f} part {a['part_acc'][0]:.3f} "
                  f"pose {a['pose_acc'][0]:.3f}", flush=True)
    names = [f"{s // 1000}k" for s in SIZES if os.path.exists(os.path.join(V3, f"test2_{s // 1000}k_seed0.npz"))]
    diffs = {f"{a}->{b}": paired_diffs(a, b) for a, b in zip(names[:-1], names[1:])}
    if len(names) > 2:
        diffs[f"{names[1]}->{names[-1]}"] = paired_diffs(names[1], names[-1])
    for k, d in diffs.items():
        for g in ("all", "tines", "palm", "neck", "handle"):
            m = d[g]
            print(f"  {k:9s} {g:7s} " + " | ".join(f"{st} {m[st][0]:+.3f} [{m[st][1]:+.3f}, {m[st][2]:+.3f}]"
                                                  for st in ("median_mm", "p90_mm", "hit90", "area90_mm2")), flush=True)
    with open(os.path.join(V3, "learning_curve.json"), "w") as f:
        json.dump(dict(per_size=res, paired_differences=diffs,
                       note="per_size: point estimate and 95 % bootstrap interval; paired_differences: later minus "
                            "earlier size on the same 5k test presses, paired bootstrap 95 % interval"), f, indent=1)
    curve_figure(res, os.path.join(V3, "learning_curve.png"))
    return res


GROUP_STATS = dict(median_mm=lambda r: np.median(r["err"]), p90_mm=lambda r: np.percentile(r["err"], 90),
                   hit90=lambda r: r["hit"].mean(), area90_mm2=lambda r: np.median(r["area"]),
                   part_acc=lambda r: r["b_ok"].mean(), pose_acc=lambda r: r["c_ok"].mean())


def _groups(r):
    g = {"all": np.ones(len(r["err"]), bool)}
    for k, p in enumerate(tc.PARTS4):
        g[p] = r["part"] == k
    g["raised tines (face up, would_move)"] = (r["part"] == 0) & (r["pose"] == 1) & r["would_move"].astype(bool)
    return g


def paired_diffs(a, b, B=1000, seed=0, s=0):
    """b minus a (e.g. "20k" -> "40k") per group with a paired bootstrap over the same test presses."""
    ra = dict(np.load(os.path.join(V3, f"test2_{a}_seed{s}.npz")))
    rb = dict(np.load(os.path.join(V3, f"test2_{b}_seed{s}.npz")))
    assert np.array_equal(ra["idx"], rb["idx"])
    rng = np.random.default_rng(seed)
    keys = ("err", "hit", "area", "b_ok", "c_ok")
    out = {}
    for g, sel in _groups(ra).items():
        n = int(sel.sum())
        if n < 5:
            continue
        A = {k: ra[k][sel] for k in keys}
        Bb = {k: rb[k][sel] for k in keys}
        boots = rng.integers(0, n, (B, n))
        d = dict(n=n)
        for name, f in GROUP_STATS.items():
            est = float(f(Bb) - f(A))
            bs = np.array([f({k: v[i] for k, v in Bb.items()}) - f({k: v[i] for k, v in A.items()}) for i in boots])
            d[name] = [est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out[g] = d
    return out


# ================================================================ seeds: mean +- std
def search_name(size, seed):
    return f"v2_{size // 1000}k_s{seed}"


def seeds_report(size, seeds=(0, 1, 2), conditions=("clean", "noise_0.05mm", "lateral_0.98")):
    """Test-set metrics (5k test set) and search-loop metrics of the same recipe with several seeds."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C = None
    k = size // 1000
    per_seed = {}
    for s in seeds:
        fn = os.path.join(V3, f"test2_{k}k_seed{s}.npz")
        if not os.path.exists(fn):
            C = C if C is not None else load3()
            test = np.nonzero(C["split"] == 4)[0]
            np.savez_compressed(fn, **per_sample(load_net(model_path(size, s), dev), C, test))
        r = dict(np.load(fn))
        m = {"contact_acc": float(r["a_ok"].mean())}
        for g, sel in _groups(r).items():
            if sel.sum() < 5:   # e.g. "raised tines (would_move)" is empty on the static-fork test2 set
                continue
            rs = {kk: r[kk][sel] for kk in ("err", "hit", "area", "b_ok", "c_ok")}
            for name, f in GROUP_STATS.items():
                m[f"{g} / {name}"] = float(f(rs))
        per_seed[s] = m
    test_stats = {key: dict(mean=float(np.mean([per_seed[s][key] for s in seeds])),
                            std=float(np.std([per_seed[s][key] for s in seeds], ddof=1)),
                            values=[per_seed[s][key] for s in seeds]) for key in per_seed[seeds[0]]}
    search = {}
    for cond in conditions:
        vals = {}
        for s in seeds:
            fn = os.path.join(V3, "search", f"episodes_{search_name(size, s)}_{cond}.json")
            if not os.path.exists(fn):
                print("missing", fn)
                continue
            eps = json.load(open(fn))
            st = v2.search_stats(eps)
            h2 = [v2.thresholds(e, 16.0) for e in eps]
            m = {"final error > 5 mm": st["final"]["err_gt5mm"],
                 "success < 16 mm2 within 2 presses": float(np.mean([h is not None and h["press"] <= 2 and h["truth_in_90"] for h in h2])),
                 "success < 16 mm2 within 3 presses": float(np.mean([h is not None and h["press"] <= 3 and h["truth_in_90"] for h in h2])),
                 "success < 16 mm2 (10 presses max)": st["<16mm2"]["success"],
                 "presses to < 16 mm2 (median)": st["<16mm2"]["presses_median"],
                 "final error median [mm]": st["final"]["err_mm_median"],
                 "final error p90 [mm]": st["final"]["err_mm_p90"],
                 "success < 4 mm2": st["<4mm2"]["success"],
                 "presses (mean)": st["final"]["presses_mean"]}
            for kk, v in m.items():
                vals.setdefault(kk, []).append(v)
        search[cond] = {kk: dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else None, values=v)
                        for kk, v in vals.items()}
    out = dict(size=size, seeds=list(seeds), test=test_stats, search=search,
               note="std with ddof=1 over the seeds; test = new 5k test set (dataset_v3/test2); search = 300 starts "
                    "from test2-0000.h5")
    with open(os.path.join(V3, f"seeds_{k}k.json"), "w") as f:
        json.dump(out, f, indent=1)
    for key in ("contact_acc", "all / median_mm", "all / p90_mm", "all / hit90", "all / area90_mm2", "all / part_acc",
                "all / pose_acc", "handle / median_mm", "handle / p90_mm", "tines / p90_mm"):
        t = test_stats[key]
        print(f"  {key:22s} {t['mean']:.3f} +- {t['std']:.3f}  ({', '.join(f'{v:.3f}' for v in t['values'])})")
    for cond, d in search.items():
        for kk, t in d.items():
            print(f"  search {cond:13s} {kk:34s} {t['mean']:.3f} +- {t['std'] if t['std'] is not None else float('nan'):.3f}  "
                  f"({', '.join(f'{v:.3f}' for v in t['values'])})")
    return out


# ================================================================ input resampling (a train / test mismatch)
INPUT_SCALES = (0.98, 0.99, 0.995, 1.0, 1.005, 1.01, 1.02)


def warp_scale(X, s):
    """Zoom the network inputs (N, H, W) about the pad centre by s (bilinear, zeros outside): the operation of the
    lateral-scale perturbation. s = 1 returns X untouched."""
    import cv2
    if s == 1.0:
        return X
    M = cv2.getRotationMatrix2D(((tc.W_IN - 1) / 2, (tc.H_IN - 1) / 2), 0, s)
    return np.stack([cv2.warpAffine(np.asarray(x, np.float32), M, (tc.W_IN, tc.H_IN), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0) for x in X])


def input_scale(chosen=0.99, scales=INPUT_SCALES, out_name="input_scale_sweep.json"):
    """augment_v2 resamples every training input (bilinear, lateral scale 0.97-1.03, never exactly 1), so no
    model has seen an input that was not resampled, and the test inputs are not. Sweep the input scale on val
    (when this cache has it) and test2 for the models of this sensor (standard: 40k seeds 0-2; mini: every
    size), write input_scale_sweep.json, and the test2 per-sample results at `chosen` (test2_*_rs{chosen}.npz).
    Other scales / out_name (e.g. --scales 0.96 ... 1.04 --out-name input_scale_sweep_wide.json): no npz."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C = load3()
    models = [(k, s, model_path(k, s)) for k in (SIZES if MINI else (40000,)) for s in (0, 1, 2)
              if os.path.exists(model_path(k, s))]
    if out_name != "input_scale_sweep.json":
        chosen = None
    scales = tuple(scales)
    out = dict(scales=list(scales), chosen=chosen, results={})
    for split, sid in (("val", 1), ("test2", 4)):
        idx = np.nonzero(C["split"] == sid)[0]
        if len(idx) == 0:
            continue
        X = C["X"][idx].astype(np.float32)
        for k, s, path in models:
            net = load_net(path, dev)
            for sc in scales:
                r = per_sample(net, C, idx, X=warp_scale(X, sc))
                out["results"][f"{k // 1000}k s{s}|{split}|{sc}"] = dict(
                    n=int(len(r["err"])), median_mm=float(np.median(r["err"])), p90_mm=float(np.percentile(r["err"], 90)),
                    hit90=float(r["hit"].mean()), area90_mm2=float(np.median(r["area"])), part_acc=float(r["b_ok"].mean()),
                    pose_acc=float(r["c_ok"].mean()), contact_acc=float(r["a_ok"].mean()))
                if split == "test2" and sc == chosen:
                    np.savez_compressed(os.path.join(V3, f"test2_{k // 1000}k_seed{s}_rs{chosen}.npz"), **r)
                e = out["results"][f"{k // 1000}k s{s}|{split}|{sc}"]
                print(f"{split:5s} {k // 1000:3d}k s{s} scale {sc:.3f}: median {e['median_mm']:.3f} p90 {e['p90_mm']:.2f} "
                      f"hit {e['hit90']:.3f} part {e['part_acc']:.3f} pose {e['pose_acc']:.3f}", flush=True)
    with open(os.path.join(V3, out_name), "w") as f:
        json.dump(out, f, indent=1)
    return out


def input_interp(B=1000, variants=None, out_name="input_interp.json"):
    """Is it the zoom or the resampling? Single press on val / test2 with the input variants of
    centre_study.INPUT_VARIANTS (as rendered, zoom 0.99, and resamplings at net zoom 1), for the models of this
    sensor. Paired bootstrap (same presses) of each variant minus 'as rendered' and minus 'zoom 0.99', on p90
    and median. -> input_interp.json"""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C = load3()
    models = [(k, s, model_path(k, s)) for k in (SIZES if MINI else (40000,)) for s in (0, 1, 2)
              if os.path.exists(model_path(k, s))]
    rng = np.random.default_rng(0)
    variants = {"as rendered": [], "zoom 0.99": [(0.99, 0, 0)], **(variants or {})} if variants else cs.INPUT_VARIANTS
    out = dict(variants={k: [list(st) for st in v] for k, v in variants.items()}, results={})
    for split, sid in (("val", 1), ("test2", 4)):
        idx = np.nonzero(C["split"] == sid)[0]
        if len(idx) == 0:
            continue
        X = C["X"][idx].astype(np.float32)
        for k, s, path in models:
            net = load_net(path, dev)
            errs = {}
            for name, steps in variants.items():
                Xv = X if not steps else np.stack([cs.warp_steps(x, steps) for x in X])
                r = per_sample(net, C, idx, X=Xv)
                errs[name] = (r["err"], r["part"])
                h = r["part"] == 3
                out["results"][f"{k // 1000}k s{s}|{split}|{name}"] = dict(
                    n=int(len(r["err"])), median_mm=float(np.median(r["err"])), p90_mm=float(np.percentile(r["err"], 90)),
                    hit90=float(r["hit"].mean()), part_acc=float(r["b_ok"].mean()), pose_acc=float(r["c_ok"].mean()),
                    handle_median_mm=float(np.median(r["err"][h])), handle_p90_mm=float(np.percentile(r["err"][h], 90)))
            n = len(errs["as rendered"][0])
            bi = rng.integers(0, n, (B, n))
            for name in variants:
                e = out["results"][f"{k // 1000}k s{s}|{split}|{name}"]
                for ref in ("as rendered", "zoom 0.99"):
                    for st, fn in (("p90", lambda a: np.percentile(a, 90, axis=1)), ("median", lambda a: np.median(a, axis=1))):
                        d = fn(errs[name][0][bi]) - fn(errs[ref][0][bi])
                        pt = float(fn(errs[name][0][None])[0] - fn(errs[ref][0][None])[0])
                        e[f"{st}_minus_{ref}"] = [pt, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
                print(f"{split:5s} {k // 1000:3d}k s{s} {name:34s} median {e['median_mm']:.3f} p90 {e['p90_mm']:.2f} | p90 - as rendered "
                      f"{e['p90_minus_as rendered'][0]:+.2f} [{e['p90_minus_as rendered'][1]:+.2f}, {e['p90_minus_as rendered'][2]:+.2f}] "
                      f"| p90 - zoom 0.99 {e['p90_minus_zoom 0.99'][0]:+.2f} [{e['p90_minus_zoom 0.99'][1]:+.2f}, "
                      f"{e['p90_minus_zoom 0.99'][2]:+.2f}] | handle p90 {e['handle_p90_mm']:.2f}", flush=True)
    with open(os.path.join(V3, out_name), "w") as f:
        json.dump(out, f, indent=1)
    return out


def input_scale_figure(path, fname="input_scale_sweep.json", title=None, shade=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(17, 5.5))
    cols = {"standard 40k": "tab:blue", "mini 20k": "tab:olive", "mini 40k": "tab:orange", "mini 100k": "tab:red",
            "mini-exact 100k": "tab:purple"}
    for d, tag in ((os.path.join(tc.OUT, "v3"), "standard"), (os.path.join(tc.OUT, "mini"), "mini"),
                   (os.path.join(tc.OUT, "mini_exact"), "mini-exact")):
        fn = os.path.join(d, fname)
        if not os.path.exists(fn):
            continue
        sw = json.load(open(fn))
        sc = sw["scales"]
        for key in sorted({k.split("|")[0] for k in sw["results"]}):
            for split, ls in (("test2", "-"), ("val", ":")):
                if f"{key}|{split}|1.0" not in sw["results"]:
                    continue
                for a, st in zip(ax, ("p90_mm", "median_mm")):
                    a.plot(sc, [sw["results"][f"{key}|{split}|{x}"][st] for x in sc], ls, marker="o", ms=3,
                           color=cols[f"{tag} {key.split()[0]}"], label=f"{tag} {key}, {split}")
    for a, lab in zip(ax, ("p90 centre error [mm]", "median centre error [mm]")):
        a.axvline(1.0, color="k", lw=0.8)
        a.set_xlabel("input zoom (1 = the image as rendered, > 1 = content enlarged)")
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="center right")
    if title is None:
        title = ("single press, fork contacts. Training zoomed every input by 0.97-1.03 (never exactly 1): the input as rendered "
                 "(zoom 1) is worse than a 1 % zoom for every such model.\nmini-exact (purple): a quarter of its training inputs "
                 "were not resampled; its dip at 1 is mostly gone")
    if shade:   # the training zoom range
        for a in ax:
            a.axvspan(0.97, 1.03, color="0.92", zorder=0)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.86, 1))
    fig.savefig(path, dpi=90)
    plt.close(fig)


def fov_compare(B=1000, resample=None):
    """Standard (32 x 24 mm) vs GelSight Mini (18.6 x 14.4 mm) field of view, each on its own 5k test set. The two
    test sets use the same seeds: the same fork placements, press targets on the fork, forces, yaws and tilts;
    only the sensor (and, scaled with it, where the target lies on the pad) differs. Standard: 40k, seeds 0-2;
    mini: every size found. Writes results_centre/mini/fov_compare.json/png; with resample=0.99, the same on the
    inputs zoomed by 0.99 (input_scale) -> fov_compare_rs0.99.json/png."""
    std_dir, mini_dir = os.path.join(tc.OUT, "v3"), os.path.join(tc.OUT, "mini")
    sfx = f"_rs{resample}" if resample else ""
    runs = [(f"standard 40k s{s}", os.path.join(std_dir, f"test2_40k_seed{s}{sfx}.npz")) for s in (0, 1, 2)]
    runs += [(f"mini {k}k s{s}", os.path.join(mini_dir, f"test2_{k}k_seed{s}{sfx}.npz")) for k in (20, 40, 100) for s in (0, 1, 2)]
    ex_dir = os.path.join(tc.OUT, "mini_exact")      # Mini, 100k, a quarter of the training inputs not resampled
    runs += [(f"mini-exact 100k s{s}", os.path.join(ex_dir, f"test2_100k_seed{s}{sfx}.npz")) for s in (0, 1, 2)]
    res = {}
    for name, fn in runs:
        if os.path.exists(fn):
            res[name] = summarise_ci(dict(np.load(fn)), B=B)
    for pre in ("standard 40k", "mini-exact 100k"):   # mean +- std over the seeds, per group and statistic
        std = [k for k in res if k.startswith(pre + " s")]
        if len(std) > 1:
            res[f"{pre}, seeds mean+-std"] = {
                g: {st: [float(np.mean([res[k][g][st][0] for k in std])), float(np.std([res[k][g][st][0] for k in std], ddof=1))]
                    for st in res[std[0]][g] if st != "n"}
                for g in res[std[0]] if g != "contact_acc"}
    search = {}
    for d, tag in ((std_dir, "standard"), (mini_dir, "mini"), (ex_dir, "mini-exact")):
        fn = os.path.join(d, "search", "search_metrics.json")
        if os.path.exists(fn):
            for k, v in json.load(open(fn)).items():
                if k.startswith("v2_40k") or k.startswith("mini_") or k.startswith("v2_100k"):
                    search[f"{tag} {k}"] = {kk: v[kk] for kk in ("<16mm2", "<4mm2", "final", "success_16mm2_within_2_presses",
                                                                  "success_16mm2_within_3_presses") if kk in v}
    out = dict(test=res, search=search, input_zoom=resample or 1.0,
               note="test: point estimate + 95 % bootstrap interval per run; each sensor on its "
               "own 5k test set with the same seeds (see fov_compare docstring)")
    os.makedirs(mini_dir, exist_ok=True)
    with open(os.path.join(mini_dir, f"fov_compare{sfx}.json"), "w") as f:
        json.dump(out, f, indent=1)
    for name in [k for k in res if "mean+-std" not in k]:
        a = res[name]["all"]
        print(f"{name:16s} median {a['median_mm'][0]:.2f} [{a['median_mm'][1]:.2f}, {a['median_mm'][2]:.2f}] p90 "
              f"{a['p90_mm'][0]:.2f} [{a['p90_mm'][1]:.2f}, {a['p90_mm'][2]:.2f}] hit {a['hit90'][0]:.3f} area "
              f"{a['area90_mm2'][0]:.0f} part {a['part_acc'][0]:.3f} pose {a['pose_acc'][0]:.3f} contact "
              f"{res[name]['contact_acc'][0]:.3f}", flush=True)
    fov_figure(res, os.path.join(mini_dir, f"fov_compare{sfx}.png"), resample)
    input_scale_figure(os.path.join(mini_dir, "input_scale.png"))
    return out


def fov_figure(res, path, resample=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [k for k in res if "mean+-std" not in k]
    groups = ["all", "tines", "palm", "neck", "handle"]
    stats = [("median_mm", "median centre error [mm]"), ("p90_mm", "p90 centre error [mm]"),
             ("hit90", "true centre inside 90 % region"), ("area90_mm2", "90 % region, median [mm2]"),
             ("part_acc", "part accuracy"), ("pose_acc", "pose accuracy")]
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    w = 0.8 / max(len(names), 1)
    for a, (st, lab) in zip(ax.ravel(), stats):
        for j, nm in enumerate(names):
            col = "tab:blue" if nm.startswith("standard") else "tab:purple" if nm.startswith("mini-exact") else \
                {"20k": "tab:olive", "40k": "tab:orange", "100k": "tab:red"}[nm.split()[1]]
            for gi, g in enumerate(groups):
                if g not in res[nm]:
                    continue
                v, lo, hi = res[nm][g][st]
                a.errorbar(gi + (j - (len(names) - 1) / 2) * w, v, yerr=[[v - lo], [hi - v]], fmt="o", color=col,
                           ms=4, capsize=2, label=nm if gi == 0 else None)
        a.set_xticks(range(len(groups)))
        a.set_xticklabels(groups)
        a.set_title(lab)
        a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("standard sensor (32 x 24 mm, blue) vs GelSight Mini field of view (18.6 x 14.4 mm): 5k test presses "
                 "each, same seeds; bars = 95 % bootstrap intervals"
                 + (f"; inputs zoomed by {resample} (resampled)" if resample else "; inputs as rendered"))
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def curve_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sizes = sorted({int(k.split("/")[0]) for k in res})
    groups = ["all", "tines", "palm", "neck", "handle", "raised tines (face up, would_move)"]
    cols = dict(all="k", tines="tab:orange", palm="tab:green", neck="tab:red", handle="tab:purple",
                **{"raised tines (face up, would_move)": "tab:brown"})
    stats = [("median_mm", "median centre error [mm]"), ("p90_mm", "p90 centre error [mm]"),
             ("area90_mm2", "90% region, median [mm2]"), ("hit90", "true centre inside 90% region"),
             ("part_acc", "part accuracy"), ("pose_acc", "pose accuracy")]
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    for a, (st, lab) in zip(ax.ravel(), stats):
        for gi, g in enumerate(groups):
            xs, ys, lo, hi = [], [], [], []
            for s in sizes:
                r = res.get(f"{s}/0", {}).get(g)
                if r:
                    xs.append(s * (1 + 0.025 * (gi - 2.5)))
                    ys.append(r[st][0]), lo.append(r[st][0] - r[st][1]), hi.append(r[st][2] - r[st][0])
            if xs:
                a.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=cols[g], capsize=3, label=g, ms=4)
        a.set_xscale("log"), a.set_xlabel("training presses"), a.set_title(lab)
        a.minorticks_off()
        a.set_xticks(sizes, [f"{round(s / 1000, 1):g}k" for s in sizes])
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(("GelSight Mini field of view: " if MINI else "") + "v2 recipe trained on "
                 + " / ".join(f"{round(s / 1000, 1):g}k" for s in sizes) + " presses (120 epochs each), new 5k test set; "
                 "bars: 95% bootstrap intervals")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ================================================================ CAD baseline: calibration + comparison
ABS_PARAMS = os.path.join(V3, "abs_params.json")
GRID_T1 = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32)


def _abs_raw_worker(args):
    Xs, threads = args
    import torch
    torch.set_num_threads(threads)
    import abs_baseline as ab
    L = ab.ABSLocalizer("cpu")
    out = []
    for X in Xs:
        t = time.time()
        r = L.predict_raw(X, GRID_T1)
        r["seconds"] = time.time() - t
        out.append(r)
    return out


_ABS_GPU = None


def abs_raw(Xs, workers=20, device=None):
    """Raw CAD matching of many presses: one GPU process (device="cuda") or CPU worker processes."""
    global _ABS_GPU
    if device == "cuda":
        import abs_baseline as ab
        import torch
        if _ABS_GPU is None:
            _ABS_GPU = ab.ABSLocalizer("cuda")
        out = []
        for X in Xs:
            t = time.time()
            r = _ABS_GPU.predict_raw(X, GRID_T1)
            torch.cuda.synchronize()
            r["seconds"] = time.time() - t
            out.append(r)
        return out
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    chunks = np.array_split(np.arange(len(Xs)), workers * 2)
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(_abs_raw_worker, [([Xs[i] for i in c], 1) for c in chunks if len(c)]))
    return sum(res, [])


def abs_maps(raws, p):
    import abs_baseline as ab
    L = ab.ABSLocalizer.__new__(ab.ABSLocalizer)               # compose() needs no templates
    out = [ab.ABSLocalizer.compose(L, r, p["tau1"], p["tau2"], p["lam"], p["eta"]) for r in raws]
    return np.array([o[0] for o in out]), np.stack([o[1] for o in out]).reshape(len(out), -1)


def abs_calibrate(n_fork=300, n_other=150):
    """Fit tau1, tau2, lam (log-likelihood of the true centre cell) and eta (contact accuracy) on
    validation presses of dataset/ (natural poses)."""
    os.makedirs(V3, exist_ok=True)
    C = load3() if MINI else tc.load_cache()           # Mini: dataset_mini's val split (natural poses only)
    rng = np.random.default_rng(7)
    va = (C["split"] == 1) & (C["natural"] if "natural" in C else True)
    fk = rng.choice(np.nonzero(va & (C["cls"] == 2))[0], n_fork, replace=False)
    ot = rng.choice(np.nonzero(va & (C["cls"] != 2))[0], n_other, replace=False)
    idx = np.concatenate([fk, ot])
    t0 = time.time()
    raws = abs_raw([C["X"][i].astype(np.float32) for i in idx])
    print(f"raw matching of {len(idx)} validation presses: {time.time() - t0:.0f} s", flush=True)
    rf = raws[:n_fork]
    v = C["vec_mm"][fk]
    cells = tc.vec_to_cell(v)
    best = None
    for t1 in GRID_T1:
        for t2 in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            for lam in (0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8):
                p = dict(tau1=t1, tau2=t2, lam=lam, eta=-2.0)
                _, H = abs_maps(rf, p)
                ll = float(np.mean(np.log(np.maximum(H[np.arange(len(H)), cells], 1e-7))))
                if best is None or ll > best[0]:
                    best = (ll, p)
    p = best[1]
    ncc = np.array([r["ncc"] if r["contact"] else -1.0 for r in raws])
    is_fork = np.r_[np.ones(n_fork, bool), np.zeros(n_other, bool)]
    etas = np.linspace(-0.5, 0.95, 59)
    acc = [float(np.mean((ncc >= e) == is_fork)) for e in etas]
    p["eta"] = float(etas[int(np.argmax(acc))])
    out = dict(params=p, val_loglik=best[0], contact_acc_val=max(acc), n_fork=n_fork, n_other=n_other,
               note=f"fitted on {'dataset_mini' if MINI else 'dataset'}/ validation presses (natural poses); "
                    "the only fitted numbers of the baseline")
    with open(ABS_PARAMS, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return out


def rgb_inputs_multi(C, idx, cond, D, workers=16, seed=99):
    """Re-reconstruct presses from their RGB with structured noise (several shards)."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    out = np.zeros((len(idx), tc.H_IN, tc.W_IN), np.float32)
    tasks, where = [], []
    for shard in sorted(set(C["shard"][idx])):
        pos = np.nonzero(C["shard"][idx] == shard)[0]
        for part in np.array_split(pos, max(1, workers // 4)):
            if len(part):
                tasks.append((shard, C["index"][idx[part]].tolist(), cond, seed))
                where.append(part)
    old = tc.DATA
    tc.DATA = D                                             # _recon_worker reads tc.DATA (set in the parent; workers re-import)
    try:
        with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_set_data, initargs=(D,)) as ex:
            for part, X in zip(where, ex.map(v2._recon_worker, tasks)):
                out[part] = X
    finally:
        tc.DATA = old
    return out


def _set_data(D):
    import train_centre
    train_centre.DATA = D


def compare_conditions():
    import cv2
    out = {"clean": lambda X: X}
    for lab, s in (("noise 0.02 mm", 0.02), ("noise 0.05 mm", 0.05), ("noise 0.1 mm", 0.1)):
        out[lab] = (lambda X, s=s: X + np.random.default_rng(123).normal(0, s, X.shape).astype(np.float32))
    for lab, s in (("blur 1 px (0.2 mm)", 1.0), ("blur 2 px (0.4 mm)", 2.0)):
        out[lab] = (lambda X, s=s: np.stack([cv2.GaussianBlur(x, (0, 0), s) for x in X]))
    for s in (0.95, 0.98, 1.02, 1.05):
        M = cv2.getRotationMatrix2D(((tc.W_IN - 1) / 2, (tc.H_IN - 1) / 2), 0, s)
        out[f"lateral scale {s}"] = (lambda X, M=M: np.stack([cv2.warpAffine(x, M, (tc.W_IN, tc.H_IN), flags=cv2.INTER_LINEAR,
                                                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0) for x in X]))
    for a in (0.03, 0.06, 0.10):
        out[f"illumination drift {a}"] = dict(drift=a)
    return out


def compare(n_per_part=250, model_name=None, tag="v2_7k"):
    """CAD baseline vs the v2 network on the same 5k-test-set fork presses (stratified by part), clean and
    perturbed; per part: median/p90 error, 90 % region hit rate and size."""
    import torch
    torch.set_num_threads(16)
    os.makedirs(V3, exist_ok=True)
    C = load3()
    rng = np.random.default_rng(11)
    test = (C["split"] == 4) & (C["cls"] == 2)
    idx = np.sort(np.concatenate([rng.choice(np.nonzero(test & (C["part"] == k))[0],
                                             min(n_per_part, int((test & (C["part"] == k)).sum())), replace=False)
                                  for k in range(4)]))
    p = json.load(open(ABS_PARAMS))["params"]
    net = load_net(model_name or os.path.join(v2.V2, "model.pt"), "cpu")
    X0 = C["X"][idx].astype(np.float32)
    res = {}
    t0 = time.time()
    for name, cond in compare_conditions().items():
        X = rgb_inputs_multi(C, idx, cond, D3) if isinstance(cond, dict) else cond(X0.copy())
        r_net = per_sample(net, C, idx, X)
        cache = os.path.join(V3 if MINI else D3, f"abs_raw_{name}.pkl")   # CAD results do not depend on the network
        if os.path.exists(cache):
            raws = pickle.load(open(cache, "rb"))
        else:
            dev = "cuda" if torch.cuda.is_available() and os.environ.get("ABS_DEVICE", "cuda") == "cuda" else None
            raws = abs_raw(list(X), device=dev)
            pickle.dump(raws, open(cache, "wb"))
        pf, H = abs_maps(raws, p)
        hm = tc.heat_metrics(H, C["vec_mm"][idx])
        r_abs = dict(err=hm["argmax_err_mm"], hit=hm["cover_90"], area=hm["area_90_mm2"], fork_called=pf >= 0.5,
                     nll=hm["nll"], cover50=hm["cover_50"], cover80=hm["cover_80"], cover95=hm["cover_95"],
                     seconds=np.array([r.get("seconds", np.nan) for r in raws]))
        r_net["fork_called"] = r_net["pa_fork"] >= 0.5
        row = {}
        for k, part in [(-1, "all")] + list(enumerate(tc.PARTS4)):
            sel = np.ones(len(idx), bool) if k < 0 else C["part"][idx] == k
            row[part] = {m: dict(median_mm=float(np.median(r["err"][sel])), p90_mm=float(np.percentile(r["err"][sel], 90)),
                                 hit90=float(r["hit"][sel].mean()), area90_mm2=float(np.median(r["area"][sel])),
                                 within5=float(np.mean(r["err"][sel] <= 5)), fork_called=float(np.mean(r["fork_called"][sel])),
                                 cover50=float(r["cover50"][sel].mean()), cover80=float(r["cover80"][sel].mean()),
                                 cover95=float(r["cover95"][sel].mean()), nll_mean=float(np.mean(r["nll"][sel])))
                         for m, r in (("net", r_net), ("cad", r_abs))}
        row["cad_seconds_per_press"] = float(np.nanmedian(r_abs["seconds"]))
        res[name] = row
        a = row["all"]
        print(f"{name:26s} net: {a['net']['median_mm']:5.1f}/{a['net']['p90_mm']:6.1f} mm hit {a['net']['hit90']:.2f} "
              f"area {a['net']['area90_mm2']:4.0f} | CAD: {a['cad']['median_mm']:5.1f}/{a['cad']['p90_mm']:6.1f} mm hit "
              f"{a['cad']['hit90']:.2f} area {a['cad']['area90_mm2']:4.0f} called fork {a['cad']['fork_called']:.2f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    with open(os.path.join(V3, f"compare_cad_vs_net_{tag}.json"), "w") as f:
        json.dump(dict(n_presses=len(idx), per_part=n_per_part, params=p, net=os.path.relpath(model_name or
                       os.path.join(v2.V2, "model.pt"), HERE), conditions=res), f, indent=1)
    compare_figure(res, os.path.join(V3, f"compare_cad_vs_net_{tag}.png"), tag)
    calibration_figure(res, os.path.join(V3, f"compare_calibration_{tag}.png"), tag)
    return res


def compare_figure(res, path, tag="v2_7k"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    conds = list(res.keys())
    parts = ["all"] + list(tc.PARTS4)
    fig, ax = plt.subplots(3, len(parts), figsize=(4.2 * len(parts), 11), sharex=True)
    x = np.arange(len(conds))
    for j, part in enumerate(parts):
        for meth, ls, col in (("net", "-", "tab:blue"), ("cad", "--", "tab:red")):
            ax[0, j].plot(x, [res[c][part][meth]["median_mm"] for c in conds], ls, color=col, marker="o", ms=3,
                          label=f"network ({tag})" if meth == "net" else "CAD baseline")
            ax[1, j].plot(x, [res[c][part][meth]["hit90"] for c in conds], ls, color=col, marker="o", ms=3)
            ax[2, j].plot(x, [res[c][part][meth]["within5"] for c in conds], ls, color=col, marker="o", ms=3)
        ax[0, j].set_yscale("log"), ax[0, j].set_title(part)
        ax[1, j].set_ylim(0, 1.02), ax[2, j].set_ylim(0, 1.02)
        ax[2, j].set_xticks(x, conds, rotation=70, fontsize=7)
    ax[0, 0].set_ylabel("median centre error [mm]"), ax[1, 0].set_ylabel("true centre in 90% region")
    ax[2, 0].set_ylabel("most likely cell within 5 mm")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("CAD analysis-by-synthesis baseline vs v2 network, same test presses (5k test set, stratified by part)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=85)
    plt.close(fig)


def calibration_figure(res, path, tag="v2_7k", conds=("clean", "noise 0.05 mm", "lateral scale 0.98",
                                                       "illumination drift 0.06")):
    """Empirical coverage of the 50/80/90/95 % most-probable regions vs the nominal level, per part."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    parts = ["all"] + list(tc.PARTS4)
    lev = [0.5, 0.8, 0.9, 0.95]
    fig, ax = plt.subplots(1, len(parts), figsize=(4 * len(parts), 4.2), sharey=True)
    styles = ["-", "--", ":", "-."]
    for j, part in enumerate(parts):
        for ci, c in enumerate([c for c in conds if c in res]):
            for meth, col in (("net", "tab:blue"), ("cad", "tab:red")):
                d = res[c][part][meth]
                if "cover50" not in d:
                    continue
                ax[j].plot(lev, [d["cover50"], d["cover80"], d["hit90"], d["cover95"]], styles[ci % 4], color=col, marker="o",
                           ms=3, label=f"{'network (' + tag + ')' if meth == 'net' else 'CAD'}, {c}")
        ax[j].plot([0.45, 1], [0.45, 1], color="0.6", lw=0.8)
        ax[j].set_title(part), ax[j].set_xlabel("nominal level of the region")
        ax[j].set_xlim(0.45, 1), ax[j].set_ylim(0, 1.02)
    ax[0].set_ylabel("true centre inside the region (fraction)")
    ax[0].legend(fontsize=6)
    fig.suptitle("Calibration of the centre maps: above the diagonal = conservative, below = overconfident")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=85)
    plt.close(fig)


# ================================================================ search loop on the new test set
class SearcherV3(v2.SearcherV2):
    """The v2 loop on starts from dataset_v3/test2-0000.h5, with a v2 network (any file) or the CAD baseline."""

    def __init__(self, perturb=None, provider="net", net_path=None, threads=2, shard="test2-0000.h5"):
        import h5py
        import torch
        cs.Searcher.__init__(self, perturb, threads)            # sim, photometric model, grid
        with h5py.File(os.path.join(D3, shard)) as f:
            self.ref = f["reference_image"][()]
            self.st = {k: f[k][()] for k in ["fork_body_pos", "fork_body_quat", "arm_cmd", "force_cmd", "com_w",
                                              "fork_pos_w", "fork_quat_w", "pose_id", "would_move", "n_tines"]}
        self.provider = provider
        if provider == "net":
            self.model = load_net(net_path, "cpu")
        else:
            import abs_baseline as ab
            self.abs = ab.ABSLocalizer("cpu", params=json.load(open(ABS_PARAMS))["params"])

    def infer_cad(self, X):
        pf, H, _ = self.abs.predict(X)
        return np.array([1 - pf, 0.0, pf]), None, None, H.reshape(tc.NC, tc.NC)


def _patch_infer(searcher):
    """episode_v2 calls cs.infer(self.model, X); with the CAD baseline, route it there (worker process only)."""
    cs.infer = lambda model, X: searcher.infer_cad(X)


def _search_worker3(args):
    idx, provider, net_path, perturb = args
    S = SearcherV3(perturb, provider, net_path)
    if provider != "net":
        S.model = None
        _patch_infer(S)
    out = []
    for i in idx:
        t = time.time()
        e = S.episode_v2(i)
        e["seconds"] = time.time() - t
        out.append(e)
    return out


def search3(runs, conditions=("clean", "noise_0.05mm", "lateral_0.98"), n=300, workers=16):
    """runs: list of (name, provider, net_path)."""
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    out_dir = os.path.join(V3, "search")
    os.makedirs(out_dir, exist_ok=True)
    C = load3()
    first = np.nonzero((C["shard"] == "test2-0000.h5") & (C["sample_type"] == 0) & (C["cls"] == 2))[0]
    pick = np.sort(np.random.default_rng(2027).choice(first, min(n, len(first)), replace=False))
    idx = C["index"][pick]
    t0 = time.time()
    for name, provider, net_path in runs:
        for cond in conditions:
            fn = os.path.join(out_dir, f"episodes_{name}_{cond}.json")
            if os.path.exists(fn):
                continue
            chunks = [list(idx[k::workers]) for k in range(workers)]
            with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
                eps = sum(ex.map(_search_worker3, [(c, provider, net_path, cs.PERTURB[cond]) for c in chunks]), [])
            eps.sort(key=lambda e: e["index"])
            for e in eps:
                g = pick[np.nonzero(idx == e["index"])[0][0]]
                e["first_part"] = tc.PARTS4[C["part"][g]]
            with open(fn, "w") as f:
                json.dump(eps, f)
            print(f"{name:14s} {cond:14s} " + v2.search_line(eps) + f" | {np.mean([e['seconds'] / e['presses'] for e in eps]):.2f} s/press "
                  f"({time.time() - t0:.0f} s)", flush=True)


def search_summary3():
    import glob
    out_dir = os.path.join(V3, "search")
    rep = {}
    for fn in sorted(glob.glob(os.path.join(out_dir, "episodes_*.json"))):
        eps = json.load(open(fn))
        name = os.path.basename(fn)[9:-5]
        st = v2.search_stats(eps)
        h2 = [v2.thresholds(e, 16.0) for e in eps]
        st["success_16mm2_within_2_presses"] = float(np.mean([h is not None and h["press"] <= 2 and h["truth_in_90"] for h in h2]))
        st["success_16mm2_within_3_presses"] = float(np.mean([h is not None and h["press"] <= 3 and h["truth_in_90"] for h in h2]))
        st["seconds_per_press"] = float(np.mean([e["seconds"] / e["presses"] for e in eps]))
        st["by_first_part"] = {p: v2.search_stats([e for e in eps if e["first_part"] == p]) for p in tc.PARTS4
                               if any(e["first_part"] == p for e in eps)}
        rep[name] = st
    with open(os.path.join(out_dir, "search_metrics.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return rep


# ================================================================ runtime per press
def runtime(n=50):
    import torch
    import abs_baseline as ab
    C = load3()
    idx = np.nonzero((C["split"] == 4) & (C["cls"] == 2))[0][:n]
    Xs = [C["X"][i].astype(np.float32) for i in idx]
    p = json.load(open(ABS_PARAMS))["params"]
    out = {}
    for dev in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
        for threads in ((1, 8) if dev == "cpu" else (8,)):
            torch.set_num_threads(threads)
            net = load_net(os.path.join(v2.V2, "model.pt"), dev)
            coords = tc.coord_channels(dev)
            ts = []
            with torch.no_grad():
                for k, X in enumerate(Xs):
                    t = time.time()
                    x = torch.tensor(X, device=dev)[None, None]
                    o = net(torch.cat([x, coords[None]], 1))
                    torch.softmax(o[3].float(), 1).cpu()
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    if k >= 3:
                        ts.append(time.time() - t)
            out[f"v2 network, {dev}" + (f", {threads} thread(s)" if dev == "cpu" else "")] = float(np.median(ts))
            L = ab.ABSLocalizer(dev, params=p)
            ts = []
            for k, X in enumerate(Xs[:20]):
                t = time.time()
                L.predict(X)
                if dev == "cuda":
                    torch.cuda.synchronize()
                if k >= 2:
                    ts.append(time.time() - t)
            out[f"CAD baseline, {dev}" + (f", {threads} thread(s)" if dev == "cpu" else "")] = float(np.median(ts))
            print({k: round(v * 1000, 1) for k, v in out.items()}, flush=True)
    with open(os.path.join(V3, "runtime.json"), "w") as f:
        json.dump(dict(seconds_per_press=out, note="median over test presses; network: one press, softmax included; "
                       "CAD: coarse FFT + refinement + map; CPU = i9-13900HX, GPU = RTX 4070 Laptop"), f, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "bank", "train", "curve", "abs-calibrate", "compare", "search",
                                    "search-summary", "runtime", "seeds", "fov-compare", "input-scale", "input-interp"])
    ap.add_argument("--runs", nargs="+", default=["v2_7k_s0:net:results_centre/v2/model.pt"],
                    help="search: name:provider:net_path (provider net or cad)")
    ap.add_argument("--conditions", nargs="+", default=["clean", "noise_0.05mm", "lateral_0.98"])
    ap.add_argument("--size", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--net", default=None, help="compare: network file (default results_centre/v2/model.pt)")
    ap.add_argument("--tag", default="v2_7k", help="compare: name of the network in the output files")
    ap.add_argument("--resample", type=float, default=None,
                    help="input-scale: the zoom whose per-sample results are kept (0.99); fov-compare: use them")
    ap.add_argument("--scales", nargs="+", type=float, default=None, help="input-scale: zooms (default INPUT_SCALES)")
    ap.add_argument("--out-name", default="input_scale_sweep.json", help="input-scale: output file in the results folder")
    ap.add_argument("--edge", action="store_true", help="input-interp: the zoom-asymmetry variants -> input_edge.json")
    args = ap.parse_args()
    if MINI and args.cmd in ("abs-calibrate", "compare", "runtime") and os.path.realpath(V3) in {
            os.path.realpath(os.path.join(tc.OUT, d)) for d in ("mini", "mini_exact", "mini_robust")}:
        raise SystemExit(f"{args.cmd} on the Mini sensor: give the CAD baseline its own CENTRE_MINI_OUT "
                         f"(e.g. results_centre/mini_cad); its templates do not depend on the sensor, its window does")
    print(f"sensor {tc.SENSOR}: data {os.path.relpath(D3, HERE)}, results {os.path.relpath(V3, HERE)}, "
          f"input {tc.W_IN} x {tc.H_IN} px" + (f", un-resampled training fraction {AUG_EXACT}" if AUG_EXACT else "")
          + (", robust augmentation" if AUG_ROBUST else ""), flush=True)
    if args.cmd == "prepare":
        prepare()
    elif args.cmd == "bank":
        build_bank3()
    elif args.cmd == "train":
        train_size(args.size, args.seed, args.epochs)
    elif args.cmd == "curve":
        curve()
    elif args.cmd == "abs-calibrate":
        abs_calibrate()
    elif args.cmd == "compare":
        compare(model_name=os.path.join(HERE, args.net) if args.net else None, tag=args.tag)
    elif args.cmd == "search":
        runs = [tuple(r.split(":")) for r in args.runs]
        search3([(a, b, os.path.join(HERE, c) if c else None) for a, b, c in runs], args.conditions)
        print(json.dumps({k: {kk: v[kk] for kk in ("success_16mm2_within_2_presses", "final", "seconds_per_press")}
                          for k, v in search_summary3().items()}, indent=1))
    elif args.cmd == "search-summary":
        print(json.dumps(search_summary3(), indent=1)[:3000])
    elif args.cmd == "runtime":
        runtime()
    elif args.cmd == "seeds":
        seeds_report(args.size, conditions=args.conditions)
    elif args.cmd == "fov-compare":
        fov_compare(resample=args.resample)
    elif args.cmd == "input-scale":
        input_scale(args.resample or 0.99, scales=args.scales or INPUT_SCALES, out_name=args.out_name)
    elif args.cmd == "input-interp":
        on = None if args.out_name == "input_scale_sweep.json" else args.out_name     # --out-name given
        if args.edge:   # the zoom asymmetry: 1.02 with and without a dimmed outer ring
            input_interp(variants=cs.EDGE_VARIANTS, out_name=on or "input_edge.json")
        else:
            input_interp(out_name=on or "input_interp.json")


if __name__ == "__main__":
    main()
