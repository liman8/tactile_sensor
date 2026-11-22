"""
Multi-object models (multi-object run, phase 3): which object is under the pad, and where is its centre of mass.

    GELSIGHT_SENSOR=mini python centre_multi.py train --model A [--seed 0]     # "multi-all": the 4 objects
    GELSIGHT_SENSOR=mini python centre_multi.py train --model B [--seed 0]     # "multi-holdout": Fork.stl never seen
    -> results_multi/multi_all/model_seed<S>.pt, results_multi/multi_holdout/model_seed<S>.pt (+ .json)

Recipe: the final fork model's (mini_robust): the same backbone (train_centre.build_model: ResNet encoder on depth +
coordinate channels, flattened features, 512-d code), the same training (120 epochs, batch 64, AdamW 2e-3,
OneCycle, bf16), the same inputs (dataset_multi X + the 5 structured-noise bank variants, drawn per sample like
centre_v3.train_size), the same augmentation (centre_v2.augment_robust: rotations, sub-pixel shifts, kernels, zoom
0.97-1.03, blur, noise, random borders), and the same test border (outer 2 px masked). One change: augment_robust's
mirror is allowed only for presses of mirror-symmetric objects (objects.json "symmetric"; not the knife).

Heads (on the shared 512-d code z):
  a  contact: hover / table / object                                       (all presses)
  id object identity: softmax over the model's objects                     (presses with object contact)
  b  part: one block of logits per object (its own parts, objects.json); the loss uses the true object's block only
     (the other objects' logits are masked out), so part classes of different objects never compete
  c  stable pose: one block of logits per object over its NATURAL stable-pose ids (objects.json); the loss uses the true
     object's block (pose addendum C: the stable-pose id replaces the earlier face up / down head)
  yaw the object's yaw in the sensor frame: the direction of its +x axis (towards the head), 72 bins of 5 deg, cross-entropy
     (pose addendum C2). Rotation / mirror augmentation transform the yaw label with the same matrix as the image and the
     centre (mirror only for mirror-symmetric objects, as before)
  d  centre-of-mass heatmap: ONE decoder shared by all objects (train_centre's decoder, 150 x 150 cells of 2 mm),
     fed z + e_obj, a learned 512-d embedding of the object hypothesis. Training uses the true object's embedding;
     inference evaluates the decoder once per hypothesis, so the search gets P(centre | press, object = h) for every
     h, and the "generic" heatmap is their mixture weighted by the identity head.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("GELSIGHT_SENSOR", "mini")
import numpy as np

import objects
import train_centre as tc
import centre_v2 as v2

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("MULTI_DATA", os.path.join(HERE, "dataset_multi"))
OUT = os.environ.get("MULTI_OUT", os.path.join(HERE, "results_multi"))       # MULTI_OUT: smoke tests elsewhere
MODELS = {"A": dict(name="multi_all", objects=["fork_stl", "fork2_stl", "knife_stl", "spoon_stl"]),
          "B": dict(name="multi_holdout", objects=["fork_stl", "knife_stl", "spoon_stl"])}
CKPT_EVERY = int(os.environ.get("CENTRE_CKPT_EVERY", "5"))
TEST_BORDER_PX = v2.TEST_BORDER_PX


def model_dir(key):
    return os.path.join(OUT, MODELS[key]["name"])


def model_path(key, seed=0):
    return os.path.join(model_dir(key), f"model_seed{seed}.pt")


# ================================================================ model
def part_layout(objs):
    """Global part index = offset[object] + the object's own part index; mask[i_obj, j_part] = part j belongs to it."""
    sizes = [len(objects.part_names(o)) for o in objs]
    off = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
    mask = np.zeros((len(objs), sum(sizes)), bool)
    for i, (o, n) in enumerate(zip(off, sizes)):
        mask[i, o:o + n] = True
    names = [f"{o}:{p}" for o in objs for p in objects.part_names(o)]
    return off, mask, names


YAW_BINS = 72
YAW_BIN = 2 * np.pi / YAW_BINS                      # 5 deg


def pose_layout(objs):
    """Global stable-pose index = offset[object] + position of the pose id in the object's natural poses."""
    ids = [natural_poses(o) for o in objs]
    off = np.concatenate([[0], np.cumsum([len(i) for i in ids])[:-1]]).astype(int)
    mask = np.zeros((len(objs), sum(len(i) for i in ids)), bool)
    for k, (o, i) in enumerate(zip(off, ids)):
        mask[k, o:o + len(i)] = True
    names = [f"{o}:{p}" for o, i in zip(objs, ids) for p in i]
    return off, mask, names, ids


def natural_poses(name):
    return list(objects.get(name)["poses"]["natural"])


def yaw_bin(u):
    """Unit vectors (..., 2) in the sensor frame -> yaw bin (0..71), angle atan2(y, x) in [0, 2 pi)."""
    import torch
    if isinstance(u, torch.Tensor):
        a = torch.remainder(torch.atan2(u[..., 1], u[..., 0]), 2 * np.pi)
        return torch.clamp((a / YAW_BIN).long(), 0, YAW_BINS - 1)
    a = np.mod(np.arctan2(u[..., 1], u[..., 0]), 2 * np.pi)
    return np.clip((a / YAW_BIN).astype(int), 0, YAW_BINS - 1)


def build_model(n_obj, n_part, n_pose=2):
    import torch
    import torch.nn as nn

    class MultiNet(nn.Module):
        """train_centre's network with an identity head, per-object part logits and an object-conditioned decoder."""

        def __init__(self):
            super().__init__()
            self.base = tc.build_model()
            self.base.head_b = nn.Linear(512, n_part)
            self.base.head_c = nn.Linear(512, n_pose)
            self.head_id = nn.Linear(512, n_obj)
            self.head_yaw = nn.Linear(512, YAW_BINS)
            self.obj_emb = nn.Embedding(n_obj, 512)
            nn.init.normal_(self.obj_emb.weight, std=0.1)
            self.n_obj = n_obj

        def code(self, x):
            b = self.base
            return b.fc(b.reduce(b.layers(b.stem(x))))

        def heat(self, z, obj):
            b = self.base
            h = b.dec(b.dec_fc(z + self.obj_emb(obj)).view(-1, 64, 10, 10))[:, 0, 5:5 + tc.NC, 5:5 + tc.NC]
            return h.reshape(len(z), tc.NC * tc.NC)

        def forward(self, x, obj=None):
            """obj: (B,) object index per sample -> heatmap (B, NC*NC); None -> every hypothesis (B, n_obj, NC*NC)."""
            b = self.base
            z = self.code(x)
            if obj is None:
                hm = torch.stack([self.heat(z, torch.full((len(z),), k, device=z.device, dtype=torch.long))
                                  for k in range(self.n_obj)], 1)
            else:
                hm = self.heat(z, obj)
            return b.head_a(z), b.head_b(z), b.head_c(z), hm, self.head_id(z), self.head_yaw(z)

    return MultiNet()


def load_net(path, device="cpu"):
    """The trained model, wrapped so that every input gets the test border treatment (outer 2 px of the depth set to 0),
    whatever code feeds it. Returns (net, meta)."""
    import torch
    meta = json.load(open(path.replace(".pt", ".json")))
    _, mask, _ = part_layout(meta["objects"])
    _, pmask, _, _ = pose_layout(meta["objects"])
    net = build_model(len(meta["objects"]), mask.shape[1], pmask.shape[1]).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()

    class Masked(torch.nn.Module):
        def __init__(self, n, px):
            super().__init__()
            self.net, self.px = n, int(px)

        def forward(self, x, obj=None):
            return self.net(v2.mask_border(x, self.px), obj)
    return Masked(net, meta.get("test_border_px", TEST_BORDER_PX)).to(device).eval(), meta


# ================================================================ data
_C = None


def load_cache():
    global _C
    if _C is None:
        _C = dict(np.load(os.path.join(DATA, "cache_multi.npz")))
        _C["cls"] = np.where(_C["contact_fork"], 2, np.where(_C["contact_table"], 1, 0))
        _C["vec_mm"] = _C["com_s"][:, :2] * 1000.0
        _C["object"] = np.array(objects.names())[_C["object_id"]]
    return _C


def rows(C, objs, split):
    return np.nonzero((C["split"] == {"train": 0, "val": 1, "test": 2}[split]) & np.isin(C["object"], objs))[0]


def yaw_vectors(C, sel):
    """The object's +x axis (towards the head) in the sensor frame, projected on the sensor x-y plane, unit length."""
    q = C["fork_quat_s"][sel]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    u = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)], 1)          # first column of R, x and y rows
    return u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)


def pose_index(C, sel, objs):
    """Global stable-pose index (pose_layout) of each press, -1 for objects outside the model or non-natural poses."""
    off, _, _, ids = pose_layout(objs)
    om = {o: i for i, o in enumerate(objs)}
    out = np.full(len(sel), -1)
    for j, (o, p) in enumerate(zip(C["object"][sel], C["pose_id"][sel])):
        k = om.get(o, -1)
        if k >= 0 and int(p) in ids[k]:
            out[j] = off[k] + ids[k].index(int(p))
    return out


def labels(C, sel, objs):
    off, _, _ = part_layout(objs)
    om = {o: i for i, o in enumerate(objs)}
    obj = np.array([om.get(o, -1) for o in C["object"][sel]])
    part = C["part"][sel]
    pg = np.where((part >= 0) & (obj >= 0), off[np.maximum(obj, 0)] + np.maximum(part, 0), -1)
    sym = np.array([objects.get(o)["symmetric"] for o in objs])
    u = yaw_vectors(C, sel)
    return dict(cls=C["cls"][sel], obj=obj, part=pg, pose=pose_index(C, sel, objs), vec=C["vec_mm"][sel],
                yaw_u=u, yaw=yaw_bin(u), mirror_ok=np.where(obj >= 0, sym[np.maximum(obj, 0)], True))


def _tensors(lab, dev):
    import torch
    return dict(cls=torch.tensor(lab["cls"], device=dev), obj=torch.tensor(lab["obj"], device=dev),
                part=torch.tensor(lab["part"], device=dev), pose=torch.tensor(lab["pose"], device=dev),
                vec=torch.tensor(lab["vec"], dtype=torch.float32, device=dev),
                yaw_u=torch.tensor(lab["yaw_u"], dtype=torch.float32, device=dev),
                yaw=torch.tensor(lab["yaw"], device=dev), mirror_ok=torch.tensor(lab["mirror_ok"], device=dev))


def losses(out, lab, vec, part_mask, pose_mask, yaw=None):
    """a contact (all), id identity, b part (true object's block), c stable pose (true object's block), d heatmap (true
    object), yaw (72 bins). Everything but a only on presses with object contact. yaw: the (augmented) yaw bins."""
    import torch
    import torch.nn.functional as F
    la, lb, lc, hm, li, ly = out
    on = lab["cls"] == 2
    z = torch.zeros((), device=la.device)
    L = dict(a=F.cross_entropy(la, lab["cls"]))
    L["id"] = F.cross_entropy(li[on], lab["obj"][on]) if on.any() else z
    pm = on & (lab["part"] >= 0)
    L["b"] = (F.cross_entropy(lb[pm].masked_fill(~part_mask[lab["obj"][pm]], -1e4), lab["part"][pm])
              if pm.any() else z)
    fm = on & (lab["pose"] >= 0)
    L["c"] = (F.cross_entropy(lc[fm].masked_fill(~pose_mask[lab["obj"][fm]], -1e4), lab["pose"][fm])
              if fm.any() else z)
    L["d"] = F.cross_entropy(hm[on], tc.heat_index(vec[on])) if on.any() else z
    yb = lab["yaw"] if yaw is None else yaw
    L["yaw"] = F.cross_entropy(ly[on], yb[on]) if on.any() else z
    return L


KEYS = ("a", "id", "b", "c", "d", "yaw")


def augment_yaw(u, draws):
    """The yaw vectors under augment_robust's target transform Mv = R(th) @ diag(mir, 1) (the same as the centre's)."""
    import torch
    th, mir = draws["th"], draws["mir"]
    c, s = torch.cos(th), torch.sin(th)
    c = torch.where(draws["untouched"], torch.round(c), c)
    s = torch.where(draws["untouched"], torch.round(s), s)
    ux = mir * u[:, 0]
    return torch.stack([c * ux - s * u[:, 1], s * ux + c * u[:, 1]], 1)


def evaluate_loss(net, X, L, coords, part_mask, pose_mask, bs=256):
    import torch
    net.eval()
    tot, n = {k: 0.0 for k in KEYS}, len(X)
    with torch.no_grad():
        for i in range(0, n, bs):
            x = torch.cat([X[i:i + bs], coords.expand(len(X[i:i + bs]), -1, -1, -1)], 1)
            lb = {k: v[i:i + bs] for k, v in L.items()}
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
                out = net(x, lb["obj"].clamp(min=0))
            Ls = losses([o.float() for o in out], lb, lb["vec"], part_mask, pose_mask)
            for k in KEYS:
                tot[k] += Ls[k].item() * len(x)
    return {k: v / n for k, v in tot.items()}


# ================================================================ training
def train(key, seed=0, epochs=120, bs=64, lr=2e-3):
    import torch
    objs = MODELS[key]["objects"]
    d = model_dir(key)
    os.makedirs(d, exist_ok=True)
    path = model_path(key, seed)
    torch.backends.cudnn.benchmark = True
    logf = open(os.path.join(d, "train.log"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C = load_cache()
    tr, va = rows(C, objs, "train"), rows(C, objs, "val")
    all_tr = np.nonzero(C["split"] == 0)[0]                  # bank row r = the r-th train row of the cache
    bank = np.load(os.path.join(DATA, "cache_multi_bank.npy"), mmap_mode="r")
    assert len(bank) == len(all_tr)
    Xc = C["X"][tr]
    Bk = np.asarray(bank[np.searchsorted(all_tr, tr)])
    n = len(tr)
    Ltr = _tensors(labels(C, tr, objs), dev)
    Lva = _tensors(labels(C, va, objs), dev)
    Xva = v2.mask_border(torch.tensor(C["X"][va].astype(np.float32), device=dev)[:, None])
    _, pmask_np, pnames = part_layout(objs)
    pmask = torch.tensor(pmask_np, device=dev)
    _, smask_np, snames, _ = pose_layout(objs)
    smask = torch.tensor(smask_np, device=dev)
    coords = tc.coord_channels(dev)
    net = build_model(len(objs), pmask_np.shape[1], smask_np.shape[1]).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(n / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    gen = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed)
    ckpt, ep0, best, hist, t_prev = path.replace(".pt", ".ckpt"), 0, None, [], 0.0
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gen.set_state(ck["gen"].to(gen.get_state().device) if hasattr(ck["gen"], "to") else ck["gen"])
        rng.bit_generator.state = ck["rng"]
        ep0, best, hist, t_prev = ck["epoch"], ck["best"], ck["hist"], ck["seconds"]
    log(f"multi {key} ({MODELS[key]['name']}): objects {objs}, {n} train / {len(va)} val presses, seed {seed}, {epochs} "
        f"epochs, robust augmentation (mirror only for {[o for o in objs if objects.get(o)['symmetric']]}), test border "
        f"{TEST_BORDER_PX} px -> {os.path.relpath(path, HERE)}" + (f" (resumed after epoch {ep0})" if ep0 else ""))
    t0 = time.time() - t_prev
    for ep in range(ep0, epochs):
        net.train()
        perm = rng.permutation(n)
        tot = {k: 0.0 for k in KEYS}
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            vsel = rng.integers(0, v2.K_VARIANTS + 1, len(idx))
            xb = np.where((vsel == v2.K_VARIANTS)[:, None, None], Xc[idx], Bk[idx, np.minimum(vsel, v2.K_VARIANTS - 1)])
            x = torch.tensor(xb.astype(np.float32), device=dev)[:, None]
            it = torch.tensor(idx, device=dev)
            lb = {k: t[it] for k, t in Ltr.items()}
            x, vr, draws = v2.augment_robust(x, lb["vec"], gen, allow_mirror=lb["mirror_ok"])
            yb = yaw_bin(augment_yaw(lb["yaw_u"], draws))
            x = torch.cat([x, coords.expand(len(x), -1, -1, -1)], 1).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=dev, dtype=torch.bfloat16):
                out = net(x, lb["obj"].clamp(min=0))
            L = losses([o.float() for o in out], lb, vr, pmask, smask, yb)
            loss = sum(L[k] for k in KEYS)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            for k in KEYS:
                tot[k] += L[k].item() * len(idx)
        vl = evaluate_loss(net, Xva, Lva, coords, pmask, smask)
        hist.append(dict(epoch=ep + 1, train={k: tot[k] / n for k in KEYS}, val=vl))
        score = sum(vl.values())
        if best is None or score < best[0]:
            best = (score, ep + 1)
            torch.save(net.state_dict(), path)
        if (ep + 1) % 5 == 0 or ep == 0:
            log(f"  [{key} s{seed}] epoch {ep + 1:3d}: train d {tot['d'] / n:.3f} yaw {tot['yaw'] / n:.3f} | val a {vl['a']:.3f} "
                f"id {vl['id']:.3f} b {vl['b']:.3f} pose {vl['c']:.3f} d {vl['d']:.3f} yaw {vl['yaw']:.3f} ({time.time() - t0:.0f} s)")
        with open(path.replace(".pt", ".heartbeat"), "w") as f:
            f.write(f"{ep + 1} {time.time() - t0:.0f}\n")
        if (ep + 1) % CKPT_EVERY == 0 and ep + 1 < epochs:
            torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), gen=gen.get_state(),
                            rng=rng.bit_generator.state, epoch=ep + 1, best=best, hist=hist, seconds=time.time() - t0),
                       ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)
    log(f"  [{key} s{seed}] best epoch {best[1]} (val loss {best[0]:.3f}), {time.time() - t0:.0f} s")
    with open(path.replace(".pt", ".json"), "w") as f:
        json.dump(dict(model=key, name=MODELS[key]["name"], objects=objs, part_names=pnames, pose_names=snames,
                       natural_poses={o: natural_poses(o) for o in objs}, yaw_bins=YAW_BINS,
                       yaw_convention="sensor frame (x = image column, y = image row), angle of the object's +x axis "
                                      "(towards the head) projected on the sensor x-y plane, atan2(y, x) in [0, 360) deg, "
                                      "bin k = [5k, 5k + 5) deg", seed=seed, n=n,
                       n_val=int(len(va)), best_epoch=best[1], best_val_loss=best[0], epochs=epochs, resumed_from=ep0,
                       aug_robust=True, mirror_objects=[o for o in objs if objects.get(o)["symmetric"]],
                       test_border_px=TEST_BORDER_PX, seconds=time.time() - t0, history=hist), f, indent=1)
    if os.path.exists(ckpt):
        os.remove(ckpt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("train")
    a.add_argument("--model", choices=list(MODELS), required=True)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.model, args.seed, args.epochs)


if __name__ == "__main__":
    main()
