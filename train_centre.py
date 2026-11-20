"""
Baseline model on dataset/ (collect_dataset.py): from the RECONSTRUCTED depth map (photometric
pipeline, 160x120) predict
    a) contact class: hover / table only / fork
    b) fork part being pressed: tines / palm / neck / handle        (fork contact only)
    c) resting pose: face down / face up                            (fork contact only)
    d) where to move: a heatmap over the sensor-frame plane (+-150 mm, 2 mm cells, softmax) for
       the vector sensor origin -> fork centre of mass             (fork contact only)

    python train_centre.py recon-error     # 0. reconstructed vs ground-truth gel depth on val
    python train_centre.py prepare         # cache: 160x120 inputs + labels -> dataset/cache_centre.npz
    python train_centre.py train           # full training set -> results_centre/model.pt
    python train_centre.py evaluate        # test split, figures, metrics.json
    python train_centre.py curve           # learning curve: 2k / 4k / 8k of the train split
    python train_centre.py all             # everything above, in order

Training uses natural poses with would_move=False only; would_move=True and propped samples are
evaluated separately. Rotation augmentation turns the depth map about the sensor origin (the
image centre) and the target vector with the same matrix (selftest: `python train_centre.py selftest`).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset")
CACHE = os.path.join(DATA, "cache_centre.npz")
OUT = os.path.join(HERE, "results_centre")

PARTS4 = ["tines", "palm", "neck", "handle"]
TINE_ROOT_MM, PALM_START_MM, NECK_START_MM = 7.0, 1.0, -15.0      # as collect_dataset.py
CLASSES = ["hover", "table", "fork"]
POSES = ["face down", "face up"]
# Sensor size (gelsight_sim.SENSORS; read from GELSIGHT_SENSOR, so spawned workers agree). This file's
# own commands work on dataset/ and need the standard sensor; centre_v3.py runs the mini pipeline.
SENSOR = os.environ.get("GELSIGHT_SENSOR", "standard")
H_IN, W_IN = {"standard": (120, 160), "mini": (72, 93)}[SENSOR]     # network input (0.2 mm / px)
MM_PER_PX_IN = 0.2
PAD_HALF_MM = (W_IN * MM_PER_PX_IN / 2, H_IN * MM_PER_PX_IN / 2)   # gel footprint: 16 x 12 / 9.3 x 7.2 mm
HALF_MM, CELL_MM = 150.0, 2.0
NC = int(2 * HALF_MM / CELL_MM)      # 150 x 150 heatmap cells
CELL_C = -HALF_MM + CELL_MM * (np.arange(NC) + 0.5)          # cell centres (mm)


# ---------------------------------------------------------------- labels
def part_from_x(x_mm):
    x = np.asarray(x_mm)
    return np.where(x >= TINE_ROOT_MM, 0, np.where(x >= PALM_START_MM, 1, np.where(x >= NECK_START_MM, 2, 3)))


def quat2mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def dominant_parts(f):
    """Per sample: the fork part carrying most of the gel-fork contact force (by the contact's
    position along the fork's x axis), -1 without fork contact."""
    gc = f["gel_contacts"]
    st, cnt = gc["start"][()], gc["count"][()]
    other, pos_w, fn = gc["other"][()], gc["pos_w"][()], gc["fn"][()]
    fq, fp = f["fork_quat_w"][()], f["fork_pos_w"][()]
    out = np.full(len(st), -1, np.int64)
    for i, (a, c) in enumerate(zip(st, cnt)):
        sel = slice(a, a + c)
        k = other[sel] == 0
        if not k.any():
            continue
        ax = quat2mat(fq[i])[:, 0]
        x = (pos_w[sel][k] - fp[i]) @ ax * 1000
        w = np.bincount(part_from_x(x), weights=np.maximum(fn[sel][k], 1e-12), minlength=4)
        out[i] = int(np.argmax(w))
    return out


def vec_to_cell(v_mm):
    v = np.asarray(v_mm, float)
    ix = np.clip(np.floor((v[..., 0] + HALF_MM) / CELL_MM), 0, NC - 1).astype(np.int64)
    iy = np.clip(np.floor((v[..., 1] + HALF_MM) / CELL_MM), 0, NC - 1).astype(np.int64)
    return iy * NC + ix


def shards(split=None):
    pat = f"{split}-*.h5" if split else "*-[0-9][0-9][0-9][0-9].h5"
    return sorted(glob.glob(os.path.join(DATA, pat)))


# ================================================================ 0. reconstruction error
def recon_error(split="val"):
    import h5py
    rows = []
    for p in shards(split):
        with h5py.File(p) as f:
            dom = dominant_parts(f)
            st, cat = f["sample_type"][()], f["pose_category"][()]
            cf = f["contact_fork"][()]
            for i in range(f.attrs["n"]):
                rec = f["depth_recon"][i].astype(np.float32)
                gt = f["depth_gel"][i].astype(np.float32)
                fork = f["depth_fork"][i].astype(np.float32)
                d = rec - gt
                con = gt > 0.05
                fk = fork > 0.05
                rows.append(dict(type=int(st[i]), cat=int(cat[i]), part=int(dom[i]), fork=bool(cf[i]),
                                 mae=float(np.abs(d).mean()),
                                 mae_contact=float(np.abs(d[con]).mean()) if con.any() else np.nan,
                                 bias_contact=float(d[con].mean()) if con.any() else np.nan,
                                 mae_fork=float(np.abs(d[fk]).mean()) if fk.any() else np.nan,
                                 mae_outside=float(np.abs(d[~con]).mean()) if (~con).any() else np.nan,
                                 peak_err=float(rec.max() - gt.max()),
                                 rel_peak_err=float((rec.max() - gt.max()) / gt.max()) if gt.max() > 0.05 else np.nan))
    keys = ["mae", "mae_contact", "bias_contact", "mae_fork", "mae_outside", "peak_err", "rel_peak_err"]

    def agg(sel):
        r = [x for x in rows if sel(x)]
        out = dict(n=len(r))
        for k in keys:
            v = np.array([x[k] for x in r], float)
            v = v[np.isfinite(v)]
            out[k] = dict(mean=float(v.mean()), median=float(np.median(v))) if len(v) else None
        return out
    types = ["on_fork", "table", "hover"]
    res = dict(split=split, definitions=dict(
        mae="mean |recon - gt| over the whole 320x240 image (mm)",
        mae_contact="... over pixels where the ground-truth gel depth > 0.05 mm",
        bias_contact="mean (recon - gt) over those pixels (negative = too shallow)",
        mae_fork="... over pixels where the fork alone presses > 0.05 mm",
        mae_outside="... over pixels without contact",
        peak_err="max(recon) - max(gt) (mm)", rel_peak_err="peak_err / max(gt)",
        part="part carrying most of the gel-fork contact force"),
        all=agg(lambda x: True),
        by_type={t: agg(lambda x, i=i: x["type"] == i) for i, t in enumerate(types)},
        fork_natural_by_part={p: agg(lambda x, i=i: x["fork"] and x["cat"] == 0 and x["part"] == i)
                              for i, p in enumerate(PARTS4)},
        fork_propped=agg(lambda x: x["fork"] and x["cat"] == 1))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "recon_error.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"reconstructed vs ground-truth gel depth, {split} split ({len(rows)} samples), mm:")
    hdr = f"{'':22s}{'n':>5s} {'MAE all':>8s} {'MAE cont':>9s} {'bias cont':>9s} {'MAE fork':>9s} {'MAE out':>8s} {'peak err':>9s} {'rel peak':>9s}"
    print(hdr)

    def line(name, a):
        g = lambda k: (a[k]["mean"] if a[k] else float("nan"))
        print(f"{name:22s}{a['n']:5d} {g('mae'):8.3f} {g('mae_contact'):9.3f} {g('bias_contact'):9.3f} "
              f"{g('mae_fork'):9.3f} {g('mae_outside'):8.3f} {g('peak_err'):9.3f} {g('rel_peak_err'):9.3f}")
    line("all", res["all"])
    for t in types:
        line(t, res["by_type"][t])
    for p in PARTS4:
        line(f"fork, {p}", res["fork_natural_by_part"][p])
    line("fork, propped (edge)", res["fork_propped"])
    return res


# ================================================================ cache
def prepare():
    import h5py
    t0 = time.time()
    cols = {k: [] for k in ["X", "split", "index", "seed", "sample_type", "pose_id", "pose_category", "would_move",
                            "contact_fork", "contact_table", "com_s", "bbox_s", "fork_pos_s", "fork_quat_s",
                            "part", "n_tines", "force_cmd", "target_part", "shard"]}
    split_ix = {"train": 0, "val": 1, "test": 2}
    for p in shards():
        with h5py.File(p) as f:
            n = int(f.attrs["n"])
            rec = f["depth_recon"][()].astype(np.float32)
            cols["X"].append(np.stack([cv2.resize(r, (W_IN, H_IN), interpolation=cv2.INTER_AREA) for r in rec])
                             .astype(np.float16))
            cols["split"].append(np.full(n, split_ix[f.attrs["split"]]))
            cols["index"].append(np.arange(n))
            cols["shard"].append(np.full(n, os.path.basename(p)))
            for k in ["seed", "sample_type", "pose_id", "pose_category", "would_move", "contact_fork",
                      "contact_table", "com_s", "bbox_s", "fork_pos_s", "fork_quat_s", "n_tines", "force_cmd",
                      "target_part"]:
                cols[k].append(f[k][()])
            cols["part"].append(dominant_parts(f))
        print(f"  {os.path.basename(p)} ({time.time() - t0:.0f} s)", flush=True)
    C = {k: np.concatenate(v) for k, v in cols.items()}
    np.savez(CACHE, **C)
    print(f"wrote {CACHE}: {len(C['seed'])} samples, {os.path.getsize(CACHE) / 1e6:.0f} MB")


def load_cache():
    C = dict(np.load(CACHE))
    C["cls"] = np.where(C["contact_fork"], 2, np.where(C["contact_table"], 1, 0))
    C["vec_mm"] = C["com_s"][:, :2] * 1000.0
    C["natural"] = C["pose_category"] == 0
    C["usable"] = C["natural"] & ~C["would_move"]          # training / main evaluation population
    return C


# ================================================================ model
def build_model():
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, cin, cout, stride=1):
            super().__init__()
            self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
            self.b1 = nn.BatchNorm2d(cout)
            self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
            self.b2 = nn.BatchNorm2d(cout)
            self.sc = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
                       if stride != 1 or cin != cout else nn.Identity())

        def forward(self, x):
            y = torch.relu(self.b1(self.c1(x)))
            return torch.relu(self.b2(self.c2(y)) + self.sc(x))

    def up(cin, cout):
        return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                             nn.Conv2d(cin, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    class Net(nn.Module):
        """ResNet-style encoder (3 input channels: depth + sensor-frame x, y coordinates, so the
        network knows WHERE in the image the contact is), flattened (not pooled) features, 3
        classification heads and a heatmap decoder."""

        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
            chans = [(32, 32, 1), (32, 64, 2), (64, 128, 2), (128, 256, 2), (256, 256, 2)]
            self.layers = nn.Sequential(*[nn.Sequential(Block(a, b, s), Block(b, b)) for a, b, s in chans])
            self.reduce = nn.Sequential(nn.Conv2d(256, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
            fh, fw = H_IN, W_IN
            for _ in range(4):                                  # 4 stride-2 stages: 120 x 160 -> 8 x 10
                fh, fw = (fh + 1) // 2, (fw + 1) // 2
            self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * fh * fw, 512), nn.ReLU(inplace=True), nn.Dropout(0.3))
            self.head_a, self.head_b, self.head_c = nn.Linear(512, 3), nn.Linear(512, 4), nn.Linear(512, 2)
            self.dec_fc = nn.Sequential(nn.Linear(512, 64 * 10 * 10), nn.ReLU(inplace=True))
            self.dec = nn.Sequential(up(64, 64), up(64, 48), up(48, 32), up(32, 16), nn.Conv2d(16, 1, 3, 1, 1))

        def forward(self, x):
            z = self.fc(self.reduce(self.layers(self.stem(x))))
            hm = self.dec(self.dec_fc(z).view(-1, 64, 10, 10))[:, 0, 5:5 + NC, 5:5 + NC]      # 160 -> 150
            return self.head_a(z), self.head_b(z), self.head_c(z), hm.reshape(len(x), NC * NC)

    return Net()


def coord_channels(device):
    import torch
    xs = (torch.arange(W_IN, device=device) - (W_IN - 1) / 2) * MM_PER_PX_IN / 16.0
    ys = (torch.arange(H_IN, device=device) - (H_IN - 1) / 2) * MM_PER_PX_IN / 12.0
    return torch.stack(torch.meshgrid(ys, xs, indexing="ij")[::-1], 0)          # (2, H, W): x, y


def augment(x, vec, gen):
    """Random rotation about the sensor origin (image centre) and random mirror (sensor x -> -x),
    applied to the depth maps x (B, 1, H, W) and to the target vectors vec (B, 2, mm) with the
    SAME matrix M. Half of the rotations are exact (0 or 180 deg: no cropped corners)."""
    import torch
    import torch.nn.functional as F
    B, dev = len(x), x.device
    th = torch.rand(B, generator=gen, device=dev) * 2 * np.pi
    exact = torch.rand(B, generator=gen, device=dev) < 0.5
    th = torch.where(exact, torch.round(th / np.pi) * np.pi, th)
    mir = torch.where(torch.rand(B, generator=gen, device=dev) < 0.5, -1.0, 1.0)
    c, s = torch.cos(th), torch.sin(th)
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)          # (B, 2, 2), acts on (x, y)
    Fm = torch.diag_embed(torch.stack([mir, torch.ones_like(mir)], -1))
    M = R @ Fm                                                                         # content transform (mm)
    S = torch.diag(torch.tensor([W_IN * MM_PER_PX_IN / 2, H_IN * MM_PER_PX_IN / 2], device=dev))  # normalised -> mm
    Sinv = torch.linalg.inv(S)
    A = Sinv @ torch.linalg.inv(M) @ S                       # output (normalised) -> input (normalised)
    theta = torch.cat([A, torch.zeros(B, 2, 1, device=dev)], 2)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    xr = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    vr = (M @ vec.unsqueeze(-1)).squeeze(-1)
    return xr, vr


def selftest():
    """A blob at a known sensor-frame point must end up at M @ point after augment()."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=dev).manual_seed(0)
    worst = 0.0
    for _ in range(20):
        p = np.array([np.random.uniform(-1, 1) * 0.5625 * PAD_HALF_MM[0],
                      np.random.uniform(-1, 1) * 0.5 * PAD_HALF_MM[1]])        # mm, inside the image (+-9 x +-6 standard)
        yy, xx = np.mgrid[0:H_IN, 0:W_IN]
        X = (xx - (W_IN - 1) / 2) * MM_PER_PX_IN
        Y = (yy - (H_IN - 1) / 2) * MM_PER_PX_IN
        img = np.exp(-((X - p[0]) ** 2 + (Y - p[1]) ** 2) / (2 * 0.6 ** 2))
        x = torch.tensor(img, dtype=torch.float32, device=dev)[None, None].repeat(8, 1, 1, 1)
        v = torch.tensor(p, dtype=torch.float32, device=dev)[None].repeat(8, 1)
        xr, vr = augment(x, v, gen)
        for k in range(8):
            im = xr[k, 0].cpu().numpy()
            w = im / im.sum()
            cx, cy = (w * X).sum(), (w * Y).sum()
            err = np.hypot(cx - vr[k, 0].item(), cy - vr[k, 1].item())
            worst = max(worst, err)
    print(f"augment selftest: blob centroid vs rotated/mirrored target vector: worst error {worst:.3f} mm "
          f"(pixel 0.2 mm) -> {'OK' if worst < 0.2 else 'FAIL'}")
    return worst


# ================================================================ training
def to_tensors(C, sel, device):
    import torch
    X = torch.tensor(C["X"][sel].astype(np.float32), device=device)[:, None]
    lab = dict(cls=torch.tensor(C["cls"][sel], device=device),
               part=torch.tensor(np.where(C["part"][sel] >= 0, C["part"][sel], 0), device=device),
               pose=torch.tensor(np.clip(C["pose_id"][sel], 0, 1), device=device),
               vec=torch.tensor(C["vec_mm"][sel], dtype=torch.float32, device=device),
               fork=torch.tensor(C["cls"][sel] == 2, device=device))
    return X, lab


def heat_index(vec):
    import torch
    ix = torch.clamp(torch.floor((vec[:, 0] + HALF_MM) / CELL_MM), 0, NC - 1).long()
    iy = torch.clamp(torch.floor((vec[:, 1] + HALF_MM) / CELL_MM), 0, NC - 1).long()
    return iy * NC + ix


def losses(out, lab, vec=None):
    import torch
    import torch.nn.functional as F
    la, lb, lc, hm = out
    vec = lab["vec"] if vec is None else vec
    fork = lab["fork"]
    L = dict(a=F.cross_entropy(la, lab["cls"]))
    if fork.any():
        L["b"] = F.cross_entropy(lb[fork], lab["part"][fork])
        L["c"] = F.cross_entropy(lc[fork], lab["pose"][fork])
        L["d"] = F.cross_entropy(hm[fork], heat_index(vec[fork]))
    else:
        L["b"] = L["c"] = L["d"] = torch.zeros((), device=la.device)
    return L


def train(C, train_sel, val_sel, out_path, epochs=60, bs=64, lr=2e-3, seed=0, log=print):
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Ltr = to_tensors(C, train_sel, dev)
    Xva, Lva = to_tensors(C, val_sel, dev)
    coords = coord_channels(dev)
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(Xtr)
    steps = epochs * int(np.ceil(n / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    gen = torch.Generator(device=dev).manual_seed(seed)
    best, hist = None, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev, generator=gen)
        tot = {k: 0.0 for k in "abcd"}
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            x, v = augment(Xtr[idx], Ltr["vec"][idx], gen)
            x = torch.cat([x, coords.expand(len(x), -1, -1, -1)], 1)
            lab = {k: t[idx] for k, t in Ltr.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
                out = model(x)
            L = losses([o.float() for o in out], lab, v)
            loss = L["a"] + L["b"] + L["c"] + L["d"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            for k in "abcd":
                tot[k] += L[k].item() * len(idx)
        vl = evaluate_loss(model, Xva, Lva, coords)
        hist.append(dict(epoch=ep + 1, train={k: tot[k] / n for k in "abcd"}, val=vl))
        score = vl["a"] + vl["b"] + vl["c"] + vl["d"]
        if best is None or score < best[0]:
            best = (score, ep + 1)
            torch.save(model.state_dict(), out_path)
        if (ep + 1) % 5 == 0 or ep == 0:
            log(f"  epoch {ep + 1:3d}: train a {tot['a'] / n:.3f} b {tot['b'] / n:.3f} c {tot['c'] / n:.3f} "
                f"d {tot['d'] / n:.3f} | val a {vl['a']:.3f} b {vl['b']:.3f} c {vl['c']:.3f} d {vl['d']:.3f} "
                f"({time.time() - t0:.0f} s)")
    log(f"  best epoch {best[1]} (val loss {best[0]:.3f}), {time.time() - t0:.0f} s")
    return dict(best_epoch=best[1], best_val_loss=best[0], history=hist, n_train=int(n), seconds=time.time() - t0)


def evaluate_loss(model, X, L, coords, bs=256):
    import torch
    model.eval()
    tot = {k: 0.0 for k in "abcd"}
    nf = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = torch.cat([X[i:i + bs], coords.expand(len(X[i:i + bs]), -1, -1, -1)], 1)
            lab = {k: t[i:i + bs] for k, t in L.items()}
            out = [o.float() for o in model(x)]
            Ls = losses(out, lab)
            tot["a"] += Ls["a"].item() * len(x)
            f = int(lab["fork"].sum())
            for k in "bcd":
                tot[k] += Ls[k].item() * f
            nf += f
    return dict(a=tot["a"] / len(X), **{k: tot[k] / max(nf, 1) for k in "bcd"})


def predict(model, C, sel, bs=256):
    """Class probabilities and full heatmaps (as float16) for the selected samples."""
    import torch
    dev = next(model.parameters()).device
    coords = coord_channels(dev)
    X = torch.tensor(C["X"][sel].astype(np.float32), device=dev)[:, None]
    model.eval()
    pa, pb, pc, hm = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = torch.cat([X[i:i + bs], coords.expand(len(X[i:i + bs]), -1, -1, -1)], 1)
            a, b, c, d = [o.float() for o in model(x)]
            pa.append(torch.softmax(a, 1).cpu().numpy())
            pb.append(torch.softmax(b, 1).cpu().numpy())
            pc.append(torch.softmax(c, 1).cpu().numpy())
            hm.append(torch.softmax(d, 1).cpu().numpy().astype(np.float32))
    return np.concatenate(pa), np.concatenate(pb), np.concatenate(pc), np.concatenate(hm)


# ================================================================ metrics
def heat_metrics(P, vec):
    """P (n, NC*NC) probabilities, vec (n, 2) mm. Argmax error, HPD coverage and area."""
    n = len(P)
    am = P.argmax(1)
    pt = np.stack([CELL_C[am % NC], CELL_C[am // NC]], 1)
    err = np.linalg.norm(pt - vec, axis=1)
    true = vec_to_cell(vec)
    srt = -np.sort(-P, 1)
    cum = np.cumsum(srt, 1)
    ptrue = P[np.arange(n), true]
    out = dict(argmax_point=pt, argmax_err_mm=err, nll=-np.log(np.maximum(ptrue, 1e-12)))
    for lev in (0.5, 0.8, 0.9, 0.95):
        k = (cum < lev).sum(1) + 1                     # cells in the lev most-probable region
        thr = srt[np.arange(n), k - 1]
        out[f"cover_{int(lev * 100)}"] = ptrue >= thr
        out[f"area_{int(lev * 100)}_mm2"] = k * CELL_MM ** 2
    # direction ("which way to move") error where the centre is > 10 mm away
    ang = np.degrees(np.abs(np.angle(np.exp(1j * (np.arctan2(pt[:, 1], pt[:, 0]) - np.arctan2(vec[:, 1], vec[:, 0]))))))
    out["dir_err_deg"] = np.where(np.linalg.norm(vec, axis=1) > 10, ang, np.nan)
    return out


def summarise(C, sel, pa, pb, pc, hmm):
    """Head accuracies and heatmap numbers for a selection (hmm = heat_metrics of its fork rows)."""
    cls, part, pose = C["cls"][sel], C["part"][sel], C["pose_id"][sel]
    fork = cls == 2
    r = dict(n=int(len(sel)), n_fork=int(fork.sum()),
             a_acc=float((pa.argmax(1) == cls).mean()) if len(cls) else None)
    if fork.any():
        r["b_acc"] = float((pb[fork].argmax(1) == part[fork]).mean())
        nat = fork & (pose <= 1)
        r["c_acc"] = float((pc[nat].argmax(1) == pose[nat]).mean()) if nat.any() else None
        e = hmm["argmax_err_mm"]
        r["d"] = dict(argmax_err_mm_median=float(np.median(e)), argmax_err_mm_mean=float(e.mean()),
                      argmax_err_mm_p90=float(np.percentile(e, 90)),
                      within_10mm=float((e <= 10).mean()), within_20mm=float((e <= 20).mean()),
                      dir_err_deg_median=float(np.nanmedian(hmm["dir_err_deg"])) if np.isfinite(hmm["dir_err_deg"]).any() else None,
                      dir_within_30deg=float((hmm["dir_err_deg"][np.isfinite(hmm["dir_err_deg"])] <= 30).mean())
                      if np.isfinite(hmm["dir_err_deg"]).any() else None,
                      dir_n=int(np.isfinite(hmm["dir_err_deg"]).sum()),
                      nll=float(hmm["nll"].mean()),
                      **{f"coverage_{l}": float(hmm[f"cover_{l}"].mean()) for l in (50, 80, 90, 95)},
                      area_90_mm2_median=float(np.median(hmm["area_90_mm2"])),
                      area_90_mm2_mean=float(hmm["area_90_mm2"].mean()))
    return r


def evaluate_split(model, C, sel_mask):
    idx = np.nonzero(sel_mask)[0]
    pa, pb, pc, P = predict(model, C, idx)
    fork = C["cls"][idx] == 2
    hm = heat_metrics(P[fork], C["vec_mm"][idx][fork])
    return idx, pa, pb, pc, P, fork, hm


def grouped(C, idx, pa, pb, pc, fork, hm, groups):
    """summaries for named boolean groups over idx (fork-only arrays in hm are aligned with fork rows)."""
    fork_pos = np.cumsum(fork) - 1
    out = {}
    for name, g in groups.items():
        gi = g[idx]
        sub_hm = {k: v[fork_pos[gi & fork]] for k, v in hm.items()}
        out[name] = summarise(C, idx[gi], pa[gi], pb[gi], pc[gi], sub_hm)
    return out


# ================================================================ figures
def fork_outline_mm(fork_pos_s, fork_quat_s, half=100.0, res=0.5):
    """Fork silhouette in the sensor-frame plane (mask on a res-mm grid over +-half)."""
    import trimesh
    from collect_dataset import rasterize
    global _MESH
    if "_MESH" not in globals():
        _MESH = trimesh.load(os.path.join(HERE, "meshes", "fork_stl", "fork_stl_visual.stl"), process=True)
    V = _MESH.vertices / 1000 @ quat2mat(fork_quat_s).T + fork_pos_s
    n = int(2 * half / res)
    P = np.c_[(V[:, 0] * 1000 + half) / res - 0.5, (V[:, 1] * 1000 + half) / res - 0.5, V[:, 2]]
    _, tid = rasterize(P, _MESH.faces, n, n)
    return tid >= 0


def sensor_x_on_fork(C, idx):
    """x (mm, along the fork, + towards the tips) of the sensor origin in the fork frame."""
    out = []
    for i in np.atleast_1d(idx):
        R = quat2mat(C["fork_quat_s"][i])
        out.append(float((R.T @ (-C["fork_pos_s"][i]))[0] * 1000))
    return np.array(out)


def pick_examples(C, per_row=4, seed=0):
    """One row per part (tines, palm, neck, handle), half face down / half face up, from the test
    split (natural, would_move=False). The handle row is from the MIDDLE of the handle (sensor
    centre 25-65 mm behind the neck), where a parallel handle would leave the direction open."""
    rng = np.random.default_rng(seed)
    test = (C["split"] == 2) & C["usable"] & (C["cls"] == 2)
    xs = np.full(len(C["seed"]), np.nan)
    ti = np.nonzero(test)[0]
    xs[ti] = sensor_x_on_fork(C, ti)
    rows = []
    for k, p in enumerate(PARTS4):
        row = []
        for pose in (0, 1):
            m = test & (C["part"] == k) & (C["pose_id"] == pose)
            if p == "handle":
                m &= (xs < -25) & (xs > -65)
            if p == "tines":
                m &= xs > 10
            cand = np.nonzero(m)[0]
            row += list(rng.choice(cand, min(per_row // 2, len(cand)), replace=False))
        rows.append(row)
    return rows, f"test presses, one row per part ({', '.join(PARTS4)}; handle row: middle of the handle)"


def pick_failures(C, per_row=4, seed=0):
    """Presses the model was not trained on: would_move=True (face-up tines) and propped (edge)."""
    rng = np.random.default_rng(seed)
    t = C["split"] == 2
    wm = np.nonzero(t & C["natural"] & C["would_move"] & (C["cls"] == 2))[0]
    pr = np.nonzero(t & (C["pose_category"] == 1) & (C["cls"] == 2))[0]
    rows = [list(rng.choice(wm, per_row, replace=False)), list(rng.choice(wm, per_row, replace=False)),
            list(rng.choice(pr, per_row, replace=False))]
    return rows, "NOT in training: rows 1-2 would_move=True (face-up fork pressed on its raised tines), row 3 propped (edge)"


def draw_examples(model, C, rows, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = np.array(sum(rows, []))
    xs = np.full(len(C["seed"]), np.nan)
    xs[pick] = sensor_x_on_fork(C, pick)
    pa, pb, pc, P = predict(model, C, pick)
    nrow, ncol = len(rows), max(len(r) for r in rows)
    fig, axes = plt.subplots(nrow, 2 * ncol, figsize=(2 * ncol * 2.55, 3.05 * nrow),
                             gridspec_kw=dict(width_ratios=[1, 1.1] * ncol), squeeze=False)
    half = 100.0
    j = 0
    for r, row in enumerate(rows):
        for c_, g in enumerate(row):
            a0, a1 = axes[r, 2 * c_], axes[r, 2 * c_ + 1]
            a0.imshow(C["X"][g].astype(np.float32), cmap="viridis", vmin=0, vmax=1.5, extent=[-16, 16, 12, -12])
            v = C["vec_mm"][g]
            pose = POSES[C["pose_id"][g]] if C["pose_id"][g] <= 1 else "propped"
            a0.set_title(f"true: {PARTS4[C['part'][g]]}, {pose}, {C['force_cmd'][g]:.1f} N, x={xs[g]:.0f} mm\n"
                         f"pred: {CLASSES[pa[j].argmax()]}, {PARTS4[pb[j].argmax()]}, {POSES[pc[j].argmax()]}",
                         fontsize=6.5)
            a0.tick_params(labelsize=5)
            d = v / max(np.linalg.norm(v), 1e-9) * 10
            a0.annotate("", xy=(d[0], d[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="r", lw=1.2))
            hm = P[j].reshape(NC, NC)
            sl = slice(int((HALF_MM - half) / CELL_MM), int((HALF_MM + half) / CELL_MM))
            a1.imshow(hm[sl, sl] ** 0.5, cmap="magma", extent=[-half, half, half, -half])
            srt = np.sort(hm.ravel())[::-1]
            thr = srt[min(np.searchsorted(np.cumsum(srt), 0.9), len(srt) - 1)]
            a1.contour(CELL_C[sl], CELL_C[sl], (hm[sl, sl] >= thr).astype(float), levels=[0.5], colors="cyan",
                       linewidths=0.7)
            sil = fork_outline_mm(C["fork_pos_s"][g], C["fork_quat_s"][g], half)
            gx = np.linspace(-half, half, sil.shape[1])
            a1.contour(gx, gx, sil.astype(float), levels=[0.5], colors="white", linewidths=0.4)
            a1.add_patch(plt.Rectangle((-16, -12), 32, 24, fill=False, ec="lime", lw=0.8))
            a1.plot(v[0], v[1], "x", color="red", ms=7, mew=1.6)
            k = P[j].argmax()
            a1.plot(CELL_C[k % NC], CELL_C[k // NC], "+", color="yellow", ms=8, mew=1.4)
            e = np.hypot(CELL_C[k % NC] - v[0], CELL_C[k // NC] - v[1])
            a1.set_title(f"err {e:.0f} mm, 90% region {int((hm >= thr).sum()) * 4} mm2", fontsize=6.5)
            a1.tick_params(labelsize=5)
            j += 1
        for c_ in range(len(row), ncol):
            axes[r, 2 * c_].axis("off")
            axes[r, 2 * c_ + 1].axis("off")
    fig.suptitle(title + ". Left: input = reconstructed depth (32x24 mm), red arrow = true direction to the centre. "
                 "Right: predicted heatmap (sqrt of prob., +-100 mm, sensor x right, y down); cyan = 90% region, red x = "
                 "true COM, yellow + = most likely cell, green = sensor pad, white = true fork outline (not an input)",
                 fontsize=8, wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=115)
    plt.close(fig)


def example_figure(model, C, path, per_row=4, seed=0):
    rows, title = pick_examples(C, per_row, seed)
    draw_examples(model, C, rows, path, title)


def calibration_figure(groups, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    lev = [50, 80, 90, 95]
    for name, r in groups.items():
        if r.get("d"):
            ax[0].plot(lev, [100 * r["d"][f"coverage_{l}"] for l in lev], "o-", label=name)
    ax[0].plot([45, 100], [45, 100], "k--", lw=0.8)
    ax[0].set_xlabel("nominal region probability [%]"), ax[0].set_ylabel("true centre inside [%]")
    ax[0].legend(fontsize=7), ax[0].set_title("heatmap calibration (test)")
    names = [n for n, r in groups.items() if r.get("d")]
    ax[1].bar(range(len(names)), [groups[n]["d"]["argmax_err_mm_median"] for n in names], alpha=0.7,
              label="median argmax error [mm]")
    ax[1].bar(range(len(names)), [np.sqrt(groups[n]["d"]["area_90_mm2_median"] / np.pi) for n in names], alpha=0.5,
              label="90% region, equivalent radius [mm]")
    ax[1].set_xticks(range(len(names)), names, rotation=30, fontsize=7)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# ================================================================ commands
def full_eval(model, C, log=print):
    test, va = C["split"] == 2, C["split"] == 1
    res = {}
    main = test & C["usable"]
    idx, pa, pb, pc, P, fork, hm = evaluate_split(model, C, main)
    groups = {"all": np.ones(len(C["seed"]), bool)}
    groups.update({f"part {p}": C["part"] == k for k, p in enumerate(PARTS4)})
    groups.update({f"pose {p}": (C["pose_id"] == k) for k, p in enumerate(POSES)})
    groups.update({f"{p}, {q}": (C["part"] == k) & (C["pose_id"] == j) for k, p in enumerate(PARTS4)
                   for j, q in enumerate(POSES)})
    groups.update({f"class {c}": C["cls"] == k for k, c in enumerate(CLASSES)})
    groups.update({f"{t} tines on pad": C["n_tines"] == t for t in range(5)})
    res["test_natural_still"] = grouped(C, idx, pa, pb, pc, fork, hm, groups)
    cm = np.zeros((3, 3), int)
    for t, p_ in zip(C["cls"][idx], pa.argmax(1)):
        cm[t, p_] += 1
    res["test_natural_still"]["confusion_a"] = dict(rows_true_cols_pred=CLASSES, matrix=cm.tolist())
    cmb = np.zeros((4, 4), int)
    for t, p_ in zip(C["part"][idx][fork], pb[fork].argmax(1)):
        cmb[t, p_] += 1
    res["test_natural_still"]["confusion_b"] = dict(rows_true_cols_pred=PARTS4, matrix=cmb.tolist())
    # separate: would_move=True (natural poses) and propped; test split, and all splits (never trained on)
    for name, m in (("would_move", C["natural"] & C["would_move"]), ("propped", C["pose_category"] == 1)):
        for scope, s in (("test", test), ("all_splits", np.ones_like(test))):
            idx2, pa2, pb2, pc2, P2, fork2, hm2 = evaluate_split(model, C, m & s)
            g = {"all": np.ones(len(C["seed"]), bool)}
            g.update({f"part {p}": C["part"] == k for k, p in enumerate(PARTS4)})
            if name == "would_move":
                g.update({f"pose {p}": (C["pose_id"] == k) for k, p in enumerate(POSES)})
            r = grouped(C, idx2, pa2, pb2, pc2, fork2, hm2, g)
            if name == "propped":
                f2 = C["cls"][idx2] == 2
                r["pose_head_says_face_up"] = float((pc2[f2].argmax(1) == 1).mean()) if f2.any() else None
                r["note"] = "pose head (c) is undefined for propped presses: it was trained on face down/up only"
            res[f"{name}_{scope}"] = r
    # val too, for reference
    idx3, pa3, pb3, pc3, P3, fork3, hm3 = evaluate_split(model, C, va & C["usable"])
    res["val_natural_still"] = grouped(C, idx3, pa3, pb3, pc3, fork3, hm3, {"all": np.ones(len(C["seed"]), bool)})
    t = res["test_natural_still"]
    log("test, natural poses, would_move=False:")
    for k in ["all", "part tines", "part palm", "part neck", "part handle", "pose face down", "pose face up",
              "class hover", "class table", "class fork"]:
        r = t[k]
        d = r.get("d") or {}
        log(f"  {k:16s} n={r['n']:4d} a={r['a_acc']:.3f} b={r.get('b_acc', float('nan')) or float('nan'):.3f} "
            f"c={r.get('c_acc', float('nan')) or float('nan'):.3f} | d: median err {d.get('argmax_err_mm_median', float('nan')):5.1f} mm, "
            f"p90 {d.get('argmax_err_mm_p90', float('nan')):5.1f}, 90%-cover {d.get('coverage_90', float('nan')):.3f}, "
            f"90% area {d.get('area_90_mm2_median', float('nan')):6.0f} mm2, dir err {d.get('dir_err_deg_median', float('nan')):5.1f} deg")
    for name in ("would_move_test", "would_move_all_splits", "propped_test", "propped_all_splits"):
        r = res[name]["all"]
        d = r.get("d") or {}
        log(f"  {name:22s} n={r['n']:4d} a={r['a_acc']:.3f} b={r.get('b_acc', float('nan')) or float('nan'):.3f} "
            f"c={r.get('c_acc') if r.get('c_acc') is not None else float('nan'):.3f} | d: median err "
            f"{d.get('argmax_err_mm_median', float('nan')):5.1f} mm, 90%-cover {d.get('coverage_90', float('nan')):.3f}, "
            f"90% area {d.get('area_90_mm2_median', float('nan')):6.0f} mm2")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["recon-error", "prepare", "selftest", "train", "evaluate", "curve", "all"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-name", default="model.pt", help="train: output file in results_centre/")
    ap.add_argument("--sizes", type=int, nargs="+", default=[2000, 4000, 8000],
                    help="curve: nominal train-split sizes (each is added to metrics.json as it finishes)")
    args = ap.parse_args()
    if SENSOR != "standard" and args.cmd != "selftest":
        raise SystemExit(f"train_centre.py works on dataset/ (standard sensor); GELSIGHT_SENSOR={SENSOR}: use centre_v3.py")
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, "train.log"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    if args.cmd in ("recon-error", "all"):
        recon_error("val")
    if args.cmd in ("prepare", "all") or (args.cmd != "recon-error" and not os.path.exists(CACHE)):
        prepare()
    if args.cmd in ("selftest", "all"):
        selftest()
    if args.cmd in ("train", "evaluate", "curve", "all"):
        import torch
        C = load_cache()
        tr, va = (C["split"] == 0) & C["usable"], (C["split"] == 1) & C["usable"]
        mpath = os.path.join(OUT, "model.pt")
        mfile = os.path.join(OUT, "metrics.json")
        metrics = json.load(open(mfile)) if os.path.exists(mfile) else {}
        if args.cmd in ("train", "all"):
            log(f"train: {tr.sum()} usable of {(C['split'] == 0).sum()} train samples (natural, would_move=False); "
                f"val {va.sum()}; {args.epochs} epochs -> {args.model_name}")
            info = train(C, tr, va, os.path.join(OUT, args.model_name), epochs=args.epochs, seed=args.seed, log=log)
            key = "training" if args.model_name == "model.pt" else f"training_{os.path.splitext(args.model_name)[0]}"
            metrics[key] = {k: v for k, v in info.items()}
            metrics[key]["epochs"] = args.epochs
            metrics["data"] = dict(train_usable=int(tr.sum()), val_usable=int(va.sum()),
                                   excluded_from_training=dict(would_move=int(((C["split"] == 0) & C["would_move"] & C["natural"]).sum()),
                                                               propped=int(((C["split"] == 0) & (C["pose_category"] == 1)).sum())))
        if args.cmd in ("evaluate", "all") and args.model_name != "model.pt":
            # another model (e.g. another training seed): metrics only, no figures
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = build_model().to(dev)
            model.load_state_dict(torch.load(os.path.join(OUT, args.model_name), map_location=dev))
            metrics[f"evaluation_{os.path.splitext(args.model_name)[0]}"] = full_eval(model, C, log)
        elif args.cmd in ("evaluate", "all"):
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model = build_model().to(dev)
            model.load_state_dict(torch.load(mpath, map_location=dev))
            metrics["evaluation"] = full_eval(model, C, log)
            example_figure(model, C, os.path.join(OUT, "examples.png"))
            rows_f, title_f = pick_failures(C)
            draw_examples(model, C, rows_f, os.path.join(OUT, "examples_not_trained.png"), title_f)
            calibration_figure({k: v for k, v in metrics["evaluation"]["test_natural_still"].items()
                                if k in ("all", "part tines", "part palm", "part neck", "part handle",
                                         "pose face down", "pose face up")},
                               os.path.join(OUT, "calibration.png"))
        if args.cmd in ("curve", "all"):
            curve = {int(k): v for k, v in metrics.get("learning_curve", {}).items()}
            first = C["index"] + 1000 * np.array([int(s.split("-")[1][:4]) for s in C["shard"]])  # index in split
            for nominal in args.sizes:
                sub = tr & (first < nominal)
                if nominal == 8000:          # = the full training set: the main model
                    path = mpath
                    info = dict(best_epoch=metrics.get("training", {}).get("best_epoch"))
                    log(f"learning curve: all {nominal} train samples -> {sub.sum()} usable (main model)")
                else:
                    # same number of optimizer steps as the full run, so the curve measures data, not steps
                    ep = int(round(args.epochs * tr.sum() / sub.sum()))
                    path = os.path.join(OUT, f"model_{nominal // 1000}k.pt")
                    log(f"learning curve: first {nominal} train samples -> {sub.sum()} usable, {ep} epochs")
                    info = train(C, sub, va, path, epochs=ep, seed=args.seed, log=log)
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                model = build_model().to(dev)
                model.load_state_dict(torch.load(path, map_location=dev))
                idx, pa, pb, pc, P, fork, hm = evaluate_split(model, C, (C["split"] == 2) & C["usable"])
                g = {"all": np.ones(len(C["seed"]), bool)}
                g.update({f"part {p}": C["part"] == k for k, p in enumerate(PARTS4)})
                r = grouped(C, idx, pa, pb, pc, fork, hm, g)
                idx_t, pa_t, pb_t, pc_t, P_t, fork_t, hm_t = evaluate_split(model, C, sub)
                r_train = grouped(C, idx_t, pa_t, pb_t, pc_t, fork_t, hm_t, {"all": np.ones(len(C["seed"]), bool)})
                curve[nominal] = dict(n_usable=int(sub.sum()), best_epoch=info["best_epoch"], test=r,
                                      train_fit=r_train["all"])
                d = r["all"]["d"]
                log(f"  {nominal}: test a {r['all']['a_acc']:.3f} b {r['all']['b_acc']:.3f} c {r['all']['c_acc']:.3f} "
                    f"| d median {d['argmax_err_mm_median']:.1f} mm, NLL {d['nll']:.2f}, 90% cover {d['coverage_90']:.3f}, "
                    f"area {d['area_90_mm2_median']:.0f} mm2 | train-fit NLL {r_train['all']['d']['nll']:.2f}")
                metrics["learning_curve"] = {str(k): v for k, v in sorted(curve.items())}
                with open(mfile, "w") as f:
                    json.dump(metrics, f, indent=1, default=float)
            metrics["learning_curve"] = {str(k): v for k, v in sorted(curve.items())}
            learning_curve_figure(curve, os.path.join(OUT, "learning_curve.png"), metrics.get("learning_curve_extra"))
        with open(mfile, "w") as f:
            json.dump(metrics, f, indent=1, default=float)


def learning_curve_figure(curve, path, extra=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = sorted(curve, key=int)
    x = [curve[n]["n_usable"] for n in ns]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for part in ["all", "part tines", "part palm", "part neck", "part handle"]:
        ax[0].plot(x, [curve[n]["test"][part]["d"]["argmax_err_mm_median"] for n in ns], "o-", label=part)
        ax[1].plot(x, [curve[n]["test"][part]["d"]["nll"] for n in ns], "o-", label=part)
    ax[1].plot(x, [curve[n]["train_fit"]["d"]["nll"] for n in ns], "k--", label="train (fit)")
    for k, lab in (("a_acc", "a contact"), ("b_acc", "b part"), ("c_acc", "c pose")):
        ax[2].plot(x, [curve[n]["test"]["all"][k] for n in ns], "o-", label=lab)
    for name, e in (extra or {}).items():        # e.g. the full set trained twice as long
        ax[0].plot(e["n_usable"], e["test"]["all"]["d"]["argmax_err_mm_median"], "k*", ms=11, label=name)
        ax[1].plot(e["n_usable"], e["test"]["all"]["d"]["nll"], "k*", ms=11, label=name)
        ax[2].plot([e["n_usable"]] * 3, [e["test"]["all"][k] for k in ("a_acc", "b_acc", "c_acc")], "k*", ms=11,
                   label=name)
    for a, t in zip(ax, ["heatmap: median argmax error [mm]", "heatmap: NLL per press (nats)", "head accuracy"]):
        a.set_xscale("log"), a.set_xlabel("training presses (usable)"), a.set_title(t), a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
