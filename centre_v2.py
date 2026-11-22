"""
v2 of the fork-centre baseline (results_centre/v2/). New assumption: the fork is CLAMPED to the
table. It lies at different places / yaws / resting poses between experiments but never moves
during a press, so would_move is irrelevant: presses on the raised tines of a face-up fork are
valid and their (static-fork) images are correct.

    python centre_v2.py bank                 # structured-noise reconstructions of the training presses (CPU)
    python centre_v2.py train [--include-edge-poses]   # augmented training -> results_centre/v2/model.pt
    python centre_v2.py evaluate             # old vs new model on the perturbation grid (by part and pose)
    python centre_v2.py tolerance            # C: lateral-scale (mm/px) tolerance per part
    python centre_v2.py calibrate            # C: simulated scale calibration (gauge, robot displacement)
    python centre_v2.py search --models new old --loops v1 v2 --conditions clean noise_0.05mm lateral_0.98

Augmentation (training only):
  * through the real pipeline: the tactile RGB is perturbed (low-frequency illumination drift,
    gel-texture pattern that moves slightly against the reference, faint leftover marker blobs)
    and reconstructed again (5 variants per training press, precomputed: `bank`);
  * on the depth input (GPU, every batch): rotation / mirror, lateral scale 0.97-1.03 (image
    only: the true centre vector does not change, as with a mm/px calibration error), Gaussian
    blur sigma 0-0.5 mm, per-pixel noise 0-0.05 mm. Hover and table-only presses get the same,
    so the contact head learns that noise is not contact.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, shift as nd_shift

import centre_study as cs
import train_centre as tc

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(tc.OUT, "v2")
BANK = os.path.join(tc.DATA, "cache_v2_bank.npy")
BANK_ROWS = os.path.join(tc.DATA, "cache_v2_bank_rows.npy")
K_VARIANTS = 5
MM_PER_PX_CAM = 0.1


# ================================================================ structured RGB noise
def smooth_field(rng, H, W):
    """Random low-frequency field, max |f| = 1 (quadratic surface)."""
    yy, xx = np.mgrid[0:H, 0:W]
    u, v = (xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)
    c = rng.normal(0, 1, 5)
    f = c[0] * u + c[1] * v + c[2] * u * u + c[3] * v * v + c[4] * u * v
    f -= f.mean()
    return f / (np.abs(f).max() + 1e-9)


def structured_noise(img, ref, rng, markers_px, drift=0.0, texture=0.0, residue=0.0):
    """Perturb a tactile RGB frame like a real sensor would, before reconstruction.
    drift: max relative gain change per channel (low-frequency field, + offset up to 50*drift grey levels);
    texture: std (grey levels) of a fine gel texture present in both frames, shifted 0.3-1 px in the image;
    residue: amplitude (grey levels) of faint dark blobs left at the (displaced) marker positions.
    Texture and residue get a random gain per colour channel (0.5-1.5): an equal change in all three
    channels looks like an albedo change to the photometric model and is almost ignored by it."""
    I = img.astype(np.float32)
    R = ref.astype(np.float32)
    H, W = I.shape[:2]
    if drift > 0:
        for c in range(3):
            a = rng.uniform(-drift, drift)
            I[..., c] = I[..., c] * (1 + a * smooth_field(rng, H, W)) + rng.uniform(-50, 50) * drift
    if texture > 0:
        T = gaussian_filter(rng.normal(0, 1, (H, W)), 1.0)
        T *= texture / (T.std() + 1e-9)
        ang, mag = rng.uniform(0, 2 * np.pi), rng.uniform(0.3, 1.0)
        gain = rng.uniform(0.5, 1.5, 3)                  # coloured LEDs: the texture shades each channel differently
        R += T[..., None] * gain
        I += nd_shift(T, (mag * np.sin(ang), mag * np.cos(ang)), order=1, mode="nearest")[..., None] * gain
    if residue > 0 and markers_px is not None:
        B = np.zeros((H, W), np.float32)
        yy, xx = np.mgrid[-8:9, -8:9]
        for (mx, my) in markers_px + rng.normal(0, 0.4, markers_px.shape):
            s = rng.uniform(1.5, 2.5)
            a = residue * rng.uniform(0.5, 1.0)
            cx, cy = int(round(mx)), int(round(my))
            g = a * np.exp(-((xx + cx - mx) ** 2 + (yy + cy - my) ** 2) / (2 * s * s))
            y0, y1, x0, x1 = cy - 8, cy + 9, cx - 8, cx + 9
            ya, yb, xa, xb = max(y0, 0), min(y1, H), max(x0, 0), min(x1, W)
            if ya < yb and xa < xb:
                B[ya:yb, xa:xb] += g[ya - y0:yb - y0, xa - x0:xb - x0]
        I -= B[..., None] * rng.uniform(0.5, 1.5, 3)       # residue is tinted differently per channel
    return np.clip(np.round(I), 0, 255).astype(np.uint8), np.clip(np.round(R), 0, 255).astype(np.uint8)


def random_variant(rng):
    """Random structured-noise amounts for one training variant."""
    return dict(drift=rng.uniform(0.01, 0.08) if rng.random() < 0.75 else 0.0,
                texture=rng.uniform(0.5, 3.0) if rng.random() < 0.75 else 0.0,
                residue=rng.uniform(3.0, 12.0) if rng.random() < 0.75 else 0.0)


def _pm():
    from fork_detector import ForkDetector
    return ForkDetector.load(cs.DETECTOR).pm


# ================================================================ bank
def _bank_worker(args):
    shard, idx, seed0 = args
    import h5py
    pm = _pm()
    with h5py.File(os.path.join(tc.DATA, shard)) as f:
        ref = f["reference_image"][()]
        rest = f["marker_rest"][()]
        out = np.zeros((len(idx), K_VARIANTS, tc.H_IN, tc.W_IN), np.float16)
        for j, i in enumerate(idx):
            img = f["img"][i]
            mk = rest + f["marker_shift"][i]
            for k in range(K_VARIANTS):
                rng = np.random.default_rng([seed0, int(i), k])
                I2, R2 = structured_noise(img, ref, rng, mk, **random_variant(rng))
                out[j, k] = cs.downsample(pm.height_map(I2, R2, MM_PER_PX_CAM)[0]).astype(np.float16)
    return shard, idx, out


def build_bank(workers=24):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    C = tc.load_cache()
    rows = np.nonzero(C["split"] == 0)[0]
    tasks = []
    for shard in sorted(set(C["shard"][rows])):
        r = rows[C["shard"][rows] == shard]
        for part in np.array_split(r, 3):
            tasks.append((shard, C["index"][part].tolist(), 2026))
    bank = np.zeros((len(rows), K_VARIANTS, tc.H_IN, tc.W_IN), np.float16)
    pos = {(C["shard"][g], int(C["index"][g])): n for n, g in enumerate(rows)}
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        for shard, idx, out in ex.map(_bank_worker, tasks):
            for j, i in enumerate(idx):
                bank[pos[(shard, i)]] = out[j]
            print(f"  {shard} {len(idx)} presses ({time.time() - t0:.0f} s)", flush=True)
    np.save(BANK, bank)
    np.save(BANK_ROWS, rows)
    print(f"wrote {BANK}: {bank.shape}, {bank.nbytes / 1e9:.2f} GB, {time.time() - t0:.0f} s")


# ================================================================ training
BLUR_BINS_PX = np.array([0.5, 1.0, 1.5, 2.0, 2.5])        # 0.1 ... 0.5 mm at 0.2 mm per input px


def augment_v2(x, vec, gen, p_exact=0.0):
    """Rotation / mirror (image AND target), lateral scale 0.97-1.03 (image only), blur 0-0.5 mm, noise 0-0.05 mm.
    p_exact > 0: that fraction of the inputs only gets a 0 / 180 degree turn and / or a mirror, with no zoom, so it is
    not resampled (every other input is; the test inputs never are). p_exact = 0 is the v2 recipe, draw for draw."""
    import torch
    import torch.nn.functional as F
    B, dev = len(x), x.device
    th = torch.rand(B, generator=gen, device=dev) * 2 * np.pi
    exact = torch.rand(B, generator=gen, device=dev) < 0.5
    th = torch.where(exact, torch.round(th / np.pi) * np.pi, th)
    mir = torch.where(torch.rand(B, generator=gen, device=dev) < 0.5, -1.0, 1.0)
    sc = 0.97 + 0.06 * torch.rand(B, generator=gen, device=dev)
    if p_exact > 0:
        ident = torch.rand(B, generator=gen, device=dev) < p_exact
        th = torch.where(ident, torch.round(th / np.pi) * np.pi, th)
        sc = torch.where(ident, torch.ones_like(sc), sc)
    c, s = torch.cos(th), torch.sin(th)
    if p_exact > 0:                                             # exactly +-1 / 0: the grid hits pixel centres
        c, s = torch.where(ident, torch.round(c), c), torch.where(ident, torch.round(s), s)
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    Fm = torch.diag_embed(torch.stack([mir, torch.ones_like(mir)], -1))
    Mv = R @ Fm                                                 # target transform
    Mi = Mv * sc[:, None, None]                                 # image transform (+ scale error)
    S = torch.diag(torch.tensor([tc.W_IN * tc.MM_PER_PX_IN / 2, tc.H_IN * tc.MM_PER_PX_IN / 2], device=dev))
    A = torch.linalg.inv(S) @ torch.linalg.inv(Mi) @ S
    grid = F.affine_grid(torch.cat([A, torch.zeros(B, 2, 1, device=dev)], 2), x.shape, align_corners=False)
    xr = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    # blur: half of the presses, sigma uniform over the bins
    b = torch.randint(0, 2 * len(BLUR_BINS_PX), (B,), generator=gen, device=dev)
    out = xr.clone()
    for k, sig in enumerate(BLUR_BINS_PX):
        sel = b == k
        if sel.any():
            r = int(np.ceil(3 * sig))
            t = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
            g = torch.exp(-t ** 2 / (2 * sig ** 2))
            g = g / g.sum()
            y = F.conv2d(F.pad(xr[sel], (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
            y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1))
            out[sel] = y
    sn = 0.05 * torch.rand(B, generator=gen, device=dev)
    out = out + sn[:, None, None, None] * torch.randn(out.shape, generator=gen, device=dev)
    return out, (Mv @ vec.unsqueeze(-1)).squeeze(-1)


TEST_BORDER_PX = 2          # the one border treatment used at test time by models trained with augment_robust
ROBUST_KERNELS = ("nearest", "bilinear", "bicubic", "area")
BORDER_TREATMENTS = ("none", "mask", "darken", "brighten", "replicate")


def mask_border(x, px=TEST_BORDER_PX):
    """The fixed test-time border treatment: the outer `px` rows and columns of the depth channel (channel 0) set
    to 0. Erases whatever an image pipeline did to the edge (0.99 dims 1 px; zoom 0.96 about 2 px)."""
    x = x.clone()
    x[:, 0, :px, :] = 0
    x[:, 0, -px:, :] = 0
    x[:, 0, :, :px] = 0
    x[:, 0, :, -px:] = 0
    return x


def input_blur(x, sigma_px):
    """Fixed Gaussian (anti-aliasing) blur of the depth channel (channel 0) at the network input, separable,
    replicate padding; the same operation in training (after the augmentation) and at test time (load_net)."""
    import torch
    import torch.nn.functional as F
    if not sigma_px:
        return x
    r = int(np.ceil(3 * sigma_px))
    t = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    g = torch.exp(-t ** 2 / (2 * float(sigma_px) ** 2))
    g = (g / g.sum()).to(x.dtype)
    d = x[:, :1]
    d = F.conv2d(F.pad(d, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
    d = F.conv2d(F.pad(d, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1))
    return torch.cat([d, x[:, 1:]], 1)


def augment_robust(x, vec, gen, p_untouched=0.25, p_test_border=0.3, params=None, allow_mirror=None):
    """Augmentation meant to make the network insensitive to the image pipeline (mini_robust).

    Geometry, per sample: rotation / mirror as in augment_v2 (image AND target); lateral zoom 0.97-1.03 (image only);
    a sub-pixel shift of 0-1 px in x and y (image AND target, so it carries no label noise); and one of four
    interpolation kernels: nearest, bilinear, bicubic, or 'area' (a box prefilter as wide as 1 / zoom when
    shrinking, then bilinear: what cv2's INTER_AREA does at these ratios). A share p_untouched is not resampled at
    all: only a 0 / 180 degree turn and / or a mirror, done by flipping (a permutation of the pixels).
    Then blur and noise as in augment_v2. Then the border, independently of the zoom: with p_test_border the test
    treatment (outer TEST_BORDER_PX px set to 0 on all sides); otherwise, for each side on its own, one of none /
    mask (0) / darken (x U(0, 1)) / brighten (x U(1, 1.5)) / replicate (the next inner row repeated), over 1-4 px.
    `params` overrides draws (tests only). allow_mirror (bool tensor, one per sample; None = all allowed, the original
    behaviour): samples where it is False are never mirrored (objects that are not mirror-symmetric, centre_multi.py);
    the random draws are the same either way. Returns the input and the target, and the draws as a dict."""
    import torch
    import torch.nn.functional as F
    B, dev = len(x), x.device
    H, W = x.shape[-2:]
    P = dict(params or {})
    u = lambda: torch.rand(B, generator=gen, device=dev)
    th = u() * 2 * np.pi
    th = torch.where(u() < 0.5, torch.round(th / np.pi) * np.pi, th)
    mir = torch.where(u() < 0.5, -1.0, 1.0)
    if allow_mirror is not None:
        mir = torch.where(allow_mirror, mir, torch.ones_like(mir))
    sc = 0.97 + 0.06 * u()
    shift = torch.stack([u(), u()], 1)                                   # px, image x (column) and y (row)
    kern = torch.randint(0, len(ROBUST_KERNELS), (B,), generator=gen, device=dev)
    untouched = u() < p_untouched
    ov = lambda k, v: torch.as_tensor(P[k], device=dev).expand_as(v).to(v.dtype) if k in P else v
    th, mir, sc, shift, kern, untouched = (ov("th", th), ov("mir", mir), ov("sc", sc), ov("shift", shift),
                                           ov("kern", kern), ov("untouched", untouched))
    th = torch.where(untouched, torch.round(th / np.pi) * np.pi, th)
    sc = torch.where(untouched, torch.ones_like(sc), sc)
    shift = torch.where(untouched[:, None], torch.zeros_like(shift), shift)
    c, s = torch.cos(th), torch.sin(th)
    c, s = torch.where(untouched, torch.round(c), c), torch.where(untouched, torch.round(s), s)
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    Fm = torch.diag_embed(torch.stack([mir, torch.ones_like(mir)], -1))
    Mv = R @ Fm                                                          # target transform
    Mi = Mv * sc[:, None, None]                                          # image transform (+ zoom)
    d_mm = shift * tc.MM_PER_PX_IN                                       # output = Mi @ input + d
    S = torch.diag(torch.tensor([W * tc.MM_PER_PX_IN / 2, H * tc.MM_PER_PX_IN / 2], device=dev))
    Si, Mii = torch.linalg.inv(S), torch.linalg.inv(Mi)
    A = Si @ Mii @ S
    t = -(Si @ Mii @ d_mm[:, :, None])
    grid = F.affine_grid(torch.cat([A, t], 2), x.shape, align_corners=False)
    xr = torch.empty_like(x)
    for k, name in enumerate(ROBUST_KERNELS):
        sel = (kern == k) & ~untouched
        if not sel.any():
            continue
        xs = x[sel]
        if name == "area":                       # box prefilter of width 1 / zoom (> 1 px only when shrinking)
            a = ((1 / sc[sel]).clamp(min=1.0) - 1) / 2                       # weight of each neighbour
            ker = torch.stack([a, 1 - 2 * a, a], 1)                          # (n, 3)
            n = len(xs)
            xp = F.pad(xs, (1, 1, 0, 0), mode="replicate")
            xs = F.conv2d(xp.reshape(1, n, H, W + 2), ker.view(n, 1, 1, 3), groups=n).reshape(n, 1, H, W)
            xp = F.pad(xs, (0, 0, 1, 1), mode="replicate")
            xs = F.conv2d(xp.reshape(1, n, H + 2, W), ker.view(n, 1, 3, 1), groups=n).reshape(n, 1, H, W)
        mode = "bilinear" if name == "area" else name
        xr[sel] = F.grid_sample(xs, grid[sel], mode=mode, padding_mode="zeros", align_corners=False)
    if untouched.any():                          # exact: mirror (x -> -x) first, then the 180 degree turn
        xu = x[untouched]
        fl = (mir[untouched] < 0)
        xu = torch.where(fl[:, None, None, None], torch.flip(xu, [3]), xu)
        r180 = c[untouched] < 0
        xu = torch.where(r180[:, None, None, None], torch.flip(xu, [2, 3]), xu)
        xr[untouched] = xu
    # blur (half of the presses) and noise, as augment_v2
    b = torch.randint(0, 2 * len(BLUR_BINS_PX), (B,), generator=gen, device=dev)
    if "blur" in P:
        b = torch.as_tensor(P["blur"], device=dev).expand(B).long()
    out = xr.clone()
    for k, sig in enumerate(BLUR_BINS_PX):
        sel = b == k
        if sel.any():
            r = int(np.ceil(3 * sig))
            tt = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
            g = torch.exp(-tt ** 2 / (2 * sig ** 2))
            g = g / g.sum()
            y = F.conv2d(F.pad(xr[sel], (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
            y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1))
            out[sel] = y
    sn = 0.05 * torch.rand(B, generator=gen, device=dev)
    if "noise" in P:
        sn = torch.as_tensor(P["noise"], device=dev, dtype=torch.float32).expand(B)
    out = out + sn[:, None, None, None] * torch.randn(out.shape, generator=gen, device=dev)
    # border, independent of the zoom. Sides: 0 top, 1 bottom, 2 left, 3 right.
    test_b = torch.rand(B, generator=gen, device=dev) < p_test_border
    treat = torch.randint(0, len(BORDER_TREATMENTS), (B, 4), generator=gen, device=dev)
    width = torch.randint(1, 5, (B, 4), generator=gen, device=dev)
    fac = torch.rand(B, 4, generator=gen, device=dev)
    if "test_border" in P:
        test_b = torch.as_tensor(P["test_border"], device=dev).expand(B).bool()
    if "treat" in P:
        treat = torch.as_tensor(P["treat"], device=dev).long().expand(B, 4)
    if "width" in P:
        width = torch.as_tensor(P["width"], device=dev).long().expand(B, 4)
    if "fac" in P:
        fac = torch.as_tensor(P["fac"], device=dev).float().expand(B, 4)
    treat = torch.where(test_b[:, None], torch.full_like(treat, 1), treat)
    width = torch.where(test_b[:, None], torch.full_like(width, TEST_BORDER_PX), width)
    rows, cols = torch.arange(H, device=dev), torch.arange(W, device=dev)
    rep = treat == 4
    lo_r = torch.where(rep[:, 0], width[:, 0], 0)
    hi_r = torch.where(rep[:, 1], H - 1 - width[:, 1], H - 1)
    lo_c = torch.where(rep[:, 2], width[:, 2], 0)
    hi_c = torch.where(rep[:, 3], W - 1 - width[:, 3], W - 1)
    ri = torch.minimum(torch.maximum(rows[None], lo_r[:, None]), hi_r[:, None])          # (B, H)
    ci = torch.minimum(torch.maximum(cols[None], lo_c[:, None]), hi_c[:, None])          # (B, W)
    out = out.gather(2, ri[:, None, :, None].expand(B, 1, H, W)).gather(3, ci[:, None, None, :].expand(B, 1, H, W))
    f = torch.ones(B, 4, device=dev)
    f = torch.where(treat == 1, 0.0, f)
    f = torch.where(treat == 2, fac, f)
    f = torch.where(treat == 3, 1 + 0.5 * fac, f)
    dist = [rows[None, :, None].expand(B, H, W), (H - 1 - rows)[None, :, None].expand(B, H, W),
            cols[None, None, :].expand(B, H, W), (W - 1 - cols)[None, None, :].expand(B, H, W)]
    fmap = torch.ones(B, H, W, device=dev)
    for sd in range(4):
        band = dist[sd] < width[:, sd, None, None]
        fmap = torch.where(band, fmap * f[:, sd, None, None], fmap)
    out = out * fmap[:, None]
    vr = (Mv @ vec.unsqueeze(-1)).squeeze(-1) + d_mm
    draws = dict(th=th, mir=mir, sc=sc, shift=shift, kern=kern, untouched=untouched, test_border=test_b, treat=treat,
                 width=width)
    return out, vr, draws


def labels_v2(C, sel, device):
    import torch
    pose = C["pose_id"][sel]
    return dict(cls=torch.tensor(C["cls"][sel], device=device),
                part=torch.tensor(np.where(C["part"][sel] >= 0, C["part"][sel], 0), device=device),
                pose=torch.tensor(np.clip(pose, 0, 1), device=device),
                pose_known=torch.tensor(pose <= 1, device=device),
                vec=torch.tensor(C["vec_mm"][sel], dtype=torch.float32, device=device),
                fork=torch.tensor(C["cls"][sel] == 2, device=device))


def losses_v2(out, lab, vec):
    import torch
    import torch.nn.functional as F
    la, lb, lc, hm = out
    fork = lab["fork"]
    L = dict(a=F.cross_entropy(la, lab["cls"]))
    z = torch.zeros((), device=la.device)
    L["b"] = F.cross_entropy(lb[fork], lab["part"][fork]) if fork.any() else z
    fp = fork & lab["pose_known"]                  # propped presses (with --include-edge-poses) have no pose label
    L["c"] = F.cross_entropy(lc[fp], lab["pose"][fp]) if fp.any() else z
    L["d"] = F.cross_entropy(hm[fork], tc.heat_index(vec[fork])) if fork.any() else z
    return L


def train_v2(include_edge=False, epochs=120, seed=0, bs=64, lr=2e-3, name="model.pt"):
    import torch
    os.makedirs(V2, exist_ok=True)
    logf = open(os.path.join(V2, "train.log"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    torch.manual_seed(seed)
    dev = "cuda"
    C = tc.load_cache()
    keep = C["natural"] | ((C["pose_category"] == 1) if include_edge else False)
    tr = np.nonzero((C["split"] == 0) & keep)[0]
    va = np.nonzero((C["split"] == 1) & C["natural"])[0]
    bank = np.load(BANK, mmap_mode="r")
    rows = np.load(BANK_ROWS)
    bpos = np.full(len(C["seed"]), -1)
    bpos[rows] = np.arange(len(rows))
    assert (bpos[tr] >= 0).all()
    Xc = C["X"][tr]                                        # clean reconstructions (float16, CPU)
    Bk = np.asarray(bank[bpos[tr]])                        # (n, K, H, W) float16, CPU
    Ltr = labels_v2(C, tr, dev)
    Xva = torch.tensor(C["X"][va].astype(np.float32), device=dev)[:, None]
    Lva = labels_v2(C, va, dev)
    coords = tc.coord_channels(dev)
    model = tc.build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(tr)
    steps = epochs * int(np.ceil(n / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    gen = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed)
    log(f"v2 train: {n} presses (natural incl. would_move{' + propped' if include_edge else ''}; "
        f"would_move ones: {int(C['would_move'][tr].sum())}), val {len(va)}, {epochs} epochs")
    best, hist, t0 = None, [], time.time()
    path = os.path.join(V2, name)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        tot = {k: 0.0 for k in "abcd"}
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            v = rng.integers(0, K_VARIANTS + 1, len(idx))       # K = clean reconstruction
            xb = np.where((v == K_VARIANTS)[:, None, None], Xc[idx], Bk[idx, np.minimum(v, K_VARIANTS - 1)])
            x = torch.tensor(xb.astype(np.float32), device=dev)[:, None]
            it = torch.tensor(idx, device=dev)
            lab = {k: t[it] for k, t in Ltr.items()}
            x, vr = augment_v2(x, lab["vec"], gen)
            x = torch.cat([x, coords.expand(len(x), -1, -1, -1)], 1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
            L = losses_v2([o.float() for o in out], lab, vr)
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
            log(f"  epoch {ep + 1:3d}: train a {tot['a'] / n:.3f} b {tot['b'] / n:.3f} c {tot['c'] / n:.3f} "
                f"d {tot['d'] / n:.3f} | val (clean) a {vl['a']:.3f} b {vl['b']:.3f} c {vl['c']:.3f} d {vl['d']:.3f} "
                f"({time.time() - t0:.0f} s)")
    log(f"  best epoch {best[1]} (clean val loss {best[0]:.3f}), {time.time() - t0:.0f} s")
    info = dict(best_epoch=best[1], best_val_loss=best[0], epochs=epochs, n_train=int(n), include_edge_poses=include_edge,
                seconds=time.time() - t0, history=hist)
    with open(os.path.join(V2, f"training_{os.path.splitext(name)[0]}.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


# ================================================================ evaluation on the perturbation grid
def load_models():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    old = cs.load_model(dev, "model.pt")
    new = tc.build_model().to(dev)
    new.load_state_dict(torch.load(os.path.join(V2, "model.pt"), map_location=dev))
    new.eval()
    return {"old (v1)": old, "new (v2)": new}


def _recon_worker(args):
    shard, idx, cond, seed = args
    import h5py
    from fork_detector import PhotometricModel
    from fork_detector import ForkDetector
    det = ForkDetector.load(cs.DETECTOR)
    pm = det.pm if "A_mult" not in cond else PhotometricModel(det.pm.A * np.asarray(cond["A_mult"]), det.pm.b)
    rgb = {k: v for k, v in cond.items() if k in ("drift", "texture", "residue")}
    with h5py.File(os.path.join(tc.DATA, shard)) as f:
        ref = f["reference_image"][()]
        rest = f["marker_rest"][()]
        out = []
        for i in idx:
            img = f["img"][i]
            if rgb:
                rng = np.random.default_rng([seed, int(i)])
                img2, ref2 = structured_noise(img, ref, rng, rest + f["marker_shift"][i], **rgb)
            else:
                img2, ref2 = img, ref
            out.append(cs.downsample(pm.height_map(img2, ref2, MM_PER_PX_CAM)[0]))
    return np.stack(out)


def rgb_condition_inputs(C, idx, cond, workers=16, seed=99):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    shard = C["shard"][idx[0]]
    assert (C["shard"][idx] == shard).all()
    parts = np.array_split(C["index"][idx], workers)
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        res = list(ex.map(_recon_worker, [(shard, p.tolist(), cond, seed) for p in parts]))
    return np.concatenate(res)


def groups_v2(C, idx):
    g = {"all": np.ones(len(idx), bool)}
    part, pose, wm = C["part"][idx], C["pose_id"][idx], C["would_move"][idx]
    for k, p in enumerate(tc.PARTS4):
        g[p] = part == k
        for j, q in enumerate(tc.POSES):
            g[f"{p}, {q}"] = (part == k) & (pose == j)
    for j, q in enumerate(tc.POSES):
        g[q] = pose == j
    g["raised tines (face up, would_move)"] = (part == 0) & (pose == 1) & wm
    g["face-up tines, not would_move"] = (part == 0) & (pose == 1) & ~wm
    return g


def score_v2(model, C, idx, X):
    """Head accuracies + heatmap numbers for groups of the selected presses, inputs X."""
    Cx = dict(C)
    Xa = C["X"].copy()
    Xa[idx] = X.astype(np.float16)
    Cx["X"] = Xa
    pa, pb, pc, P = tc.predict(model, Cx, idx, bs=128)
    fork = C["cls"][idx] == 2
    hm = tc.heat_metrics(P[fork], C["vec_mm"][idx][fork])
    out = dict(a_acc=float((pa.argmax(1) == C["cls"][idx]).mean()),
               a_hover_called_fork=float((pa[C["cls"][idx] == 0].argmax(1) == 2).mean()) if (C["cls"][idx] == 0).any() else None,
               a_table_called_fork=float((pa[C["cls"][idx] == 1].argmax(1) == 2).mean()) if (C["cls"][idx] == 1).any() else None)
    gf = groups_v2(C, idx[fork])
    for name, sel in gf.items():
        if not sel.any():
            continue
        e = hm["argmax_err_mm"][sel]
        pidx = idx[fork][sel]
        out[name] = dict(n=int(sel.sum()), median_mm=float(np.median(e)), p90_mm=float(np.percentile(e, 90)),
                         hit90=float(hm["cover_90"][sel].mean()), area90_mm2=float(np.median(hm["area_90_mm2"][sel])),
                         within10=float((e <= 10).mean()),
                         b_acc=float((pb[fork][sel].argmax(1) == C["part"][pidx]).mean()),
                         c_acc=float((pc[fork][sel].argmax(1) == C["pose_id"][pidx]).mean())
                         if (C["pose_id"][pidx] <= 1).all() else None)
    return out


def depth_conditions():
    out = {}
    for name, levels in cs.perturbations().items():
        out[name] = [(lab, f) for lab, _, f in levels]
    return out


def rgb_conditions():
    out = {"photometric_A": [(lab, dict(A_mult=M.tolist())) for lab, M in cs.photometric_variants()],
           "illumination_drift": [(f"{a}", dict(drift=a)) for a in (0.03, 0.06, 0.10)],
           "gel_texture_grey": [(f"{a}", dict(texture=a)) for a in (1.0, 2.0, 4.0)],
           "marker_residue_grey": [(f"{a}", dict(residue=a)) for a in (4.0, 8.0, 12.0)],
           "combined": [("drift .06 + texture 3 + residue 9", dict(drift=0.06, texture=3.0, residue=9.0))]}
    return out


def evaluate_v2():
    os.makedirs(V2, exist_ok=True)
    C = tc.load_cache()
    models = load_models()
    test = np.nonzero((C["split"] == 2) & C["natural"])[0]           # all natural test presses (incl. would_move)
    propped = np.nonzero((C["split"] == 2) & (C["pose_category"] == 1))[0]
    X0 = C["X"][test].astype(np.float32)
    res = {m: {} for m in models}
    t0 = time.time()
    for ctype, levels in depth_conditions().items():
        for lab, f in levels:
            X = f(X0.copy())
            for m, model in models.items():
                res[m].setdefault(ctype, {})[lab] = score_v2(model, C, test, X)
            print_line(res, ctype, lab, t0)
    for ctype, levels in rgb_conditions().items():
        for lab, cond in levels:
            X = rgb_condition_inputs(C, test, cond)
            for m, model in models.items():
                res[m].setdefault(ctype, {})[lab] = score_v2(model, C, test, X)
            print_line(res, ctype, lab, t0)
    # propped (edge) presses, clean: never trained on (unless --include-edge-poses)
    for m, model in models.items():
        res[m]["propped_clean"] = score_v2(model, C, propped, C["X"][propped].astype(np.float32))
    with open(os.path.join(V2, "robustness.json"), "w") as f:
        json.dump(res, f, indent=1)
    robustness_figure_v2(res, os.path.join(V2, "robustness.png"))
    return res


def print_line(res, ctype, lab, t0):
    s = []
    for m in res:
        r = res[m][ctype][lab]
        s.append(f"{m}: a {r['a_acc']:.2f} | " + " ".join(f"{p} {r[p]['median_mm']:.1f}/{r[p]['hit90']:.2f}"
                                                        for p in ["tines", "palm", "neck", "handle"] if p in r)
                 + f" | raised {r['raised tines (face up, would_move)']['median_mm']:.1f}/"
                   f"{r['raised tines (face up, would_move)']['hit90']:.2f}")
    print(f"{ctype:20s} {lab:>12s}  " + "  ||  ".join(s) + f"  ({time.time() - t0:.0f} s)", flush=True)


def robustness_figure_v2(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    types = ["blur_sigma_px", "noise_mm", "lateral_scale", "depth_scale", "illumination_drift", "gel_texture_grey",
             "marker_residue_grey", "photometric_A"]
    cols = dict(tines="tab:orange", palm="tab:green", neck="tab:red", handle="tab:purple",
                **{"raised tines (face up, would_move)": "tab:brown"})
    fig, ax = plt.subplots(3, len(types), figsize=(4.0 * len(types), 11))
    for j, t in enumerate(types):
        labs = list(res["new (v2)"][t].keys())
        x = np.arange(len(labs))
        for m, ls in (("old (v1)", "--"), ("new (v2)", "-")):
            for p, c in cols.items():
                ax[0, j].plot(x, [res[m][t][l][p]["median_mm"] for l in labs], ls, color=c, marker="o", ms=3,
                              label=f"{p} {m}" if j == 0 else None)
                ax[1, j].plot(x, [res[m][t][l][p]["hit90"] for l in labs], ls, color=c, marker="o", ms=3)
            ax[2, j].plot(x, [res[m][t][l]["a_acc"] for l in labs], ls, color="tab:blue", marker="s", ms=3,
                          label=f"a contact {m}")
            ax[2, j].plot(x, [res[m][t][l]["all"]["b_acc"] for l in labs], ls, color="tab:gray", marker="^", ms=3,
                          label=f"b part {m}")
        for i in range(3):
            ax[i, j].set_xticks(x, labs, fontsize=7, rotation=20)
        ax[0, j].set_yscale("log"), ax[0, j].axhline(5, color="gray", ls=":", lw=0.8)
        ax[1, j].set_ylim(0, 1.02), ax[1, j].axhline(0.9, color="gray", ls=":", lw=0.8)
        ax[2, j].set_ylim(0, 1.02)
        ax[0, j].set_title(t)
    ax[0, 0].set_ylabel("median centre error [mm]")
    ax[1, 0].set_ylabel("true centre inside 90% region")
    ax[2, 0].set_ylabel("head accuracy")
    ax[0, 0].legend(fontsize=6)
    ax[2, 0].legend(fontsize=6)
    fig.suptitle("old (v1, dashed) vs new (v2, solid) model on the test split, all natural presses incl. raised "
                 "face-up tines; per fork part")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=80)
    plt.close(fig)


# ================================================================ C: scale tolerance + calibration
def tolerance():
    C = tc.load_cache()
    models = load_models()
    test = np.nonzero((C["split"] == 2) & C["natural"])[0]
    X0 = C["X"][test].astype(np.float32)
    scales = np.round(np.arange(0.90, 1.1001, 0.01), 2)
    res = {m: {} for m in models}
    for s in scales:
        M = cv2.getRotationMatrix2D(((tc.W_IN - 1) / 2, (tc.H_IN - 1) / 2), 0, float(s))
        X = np.stack([cv2.warpAffine(x, M, (tc.W_IN, tc.H_IN), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0) for x in X0]) if s != 1 else X0
        for m, model in models.items():
            res[m][f"{s:.2f}"] = score_v2(model, C, test, X)
        print(f"scale {s:.2f}: " + " || ".join(
            f"{m}: " + " ".join(f"{p} {res[m][f'{s:.2f}'][p]['median_mm']:.1f}/{res[m][f'{s:.2f}'][p]['hit90']:.2f}"
                                for p in tc.PARTS4) for m in models), flush=True)
    # tolerance: largest |s - 1| on BOTH sides with hit rate >= 0.9 and median error <= 2 mm
    tol = {}
    for m in models:
        tol[m] = {}
        for p in list(tc.PARTS4) + [f"{p}, {q}" for p in tc.PARTS4 for q in tc.POSES] + ["raised tines (face up, would_move)"]:
            ok = {float(s): (res[m][f"{s:.2f}"].get(p) or {}).get("hit90", 0) >= 0.9 and
                  (res[m][f"{s:.2f}"].get(p) or {}).get("median_mm", 99) <= 2.0 for s in scales}
            t = 0.0
            for d in np.round(np.arange(0.01, 0.1001, 0.01), 2):
                if ok.get(round(1 - d, 2), False) and ok.get(round(1 + d, 2), False):
                    t = float(d)
                else:
                    break
            tol[m][p] = t
    out = dict(criterion="largest d such that scales 1-d ... 1+d all keep the 90%-region hit rate >= 0.9 and the "
                         "median centre error <= 2 mm (steps of 1 %)", tolerance=tol, per_scale=res)
    with open(os.path.join(V2, "scale_tolerance.json"), "w") as f:
        json.dump(out, f, indent=1)
    tolerance_figure(res, scales, os.path.join(V2, "scale_tolerance.png"))
    print(json.dumps(tol, indent=1))
    return out


def tolerance_figure(res, scales, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    cols = dict(tines="tab:orange", palm="tab:green", neck="tab:red", handle="tab:purple")
    for m, ls in (("old (v1)", "--"), ("new (v2)", "-")):
        for p, c in cols.items():
            ax[0].plot((scales - 1) * 100, [res[m][f"{s:.2f}"][p]["hit90"] for s in scales], ls, color=c, marker="o",
                       ms=3, label=f"{p} {m}")
            ax[1].plot((scales - 1) * 100, [res[m][f"{s:.2f}"][p]["median_mm"] for s in scales], ls, color=c,
                       marker="o", ms=3, label=f"{p} {m}")
    ax[0].axhline(0.9, color="gray", ls=":"), ax[1].axhline(2.0, color="gray", ls=":")
    ax[0].set_ylabel("true centre inside 90% region"), ax[1].set_ylabel("median centre error [mm]")
    ax[1].set_yscale("log")
    for a in ax:
        a.set_xlabel("mm/px error [%] (lateral scale of the input - 1)")
        a.legend(fontsize=7)
    fig.suptitle("scale tolerance per part: old (dashed) vs new (solid) model, test split, all natural presses")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


GAUGE_SPACING_MM = 10.0
GAUGE_BALL_R_MM = 1.5


def add_gauge(spec):
    """Calibration gauge: two 3 mm balls, centres exactly GAUGE_SPACING_MM apart, on the table."""
    import mujoco
    b = spec.worldbody.add_body()
    b.name = "calgauge"
    for k, sx in enumerate((-GAUGE_SPACING_MM / 2, GAUGE_SPACING_MM / 2)):
        g = b.add_geom()
        g.name = f"calgauge_ball{k}"
        g.type = mujoco.mjtGeom.mjGEOM_SPHERE
        g.size = [GAUGE_BALL_R_MM / 1000, 0, 0]
        g.pos = [sx / 1000, 0, GAUGE_BALL_R_MM / 1000]
        g.group, g.contype, g.conaffinity, g.condim = 5, 0, 0, 3
        g.friction = [0.8, 0.005, 0.0001]
        g.solref = [0.001, 1]
        g.solimp = [0.99, 0.999, 0.0002, 0.5, 2]
        g.density = 7850
    s = b.add_site()
    s.name = "calgauge_face"
    s.pos = [0, 0, 2 * GAUGE_BALL_R_MM / 1000]
    s.quat = [0, 1, 0, 0]
    s.group = 4


def _contact_centroids(h, n, with_area=False):
    """Weighted centroids (col, row) of the n largest contact blobs of a depth map (mm)."""
    m = (h > 0.3 * h.max()).astype(np.uint8)
    k, lab = cv2.connectedComponents(m)
    blobs = sorted(((lab == i).sum(), i) for i in range(1, k))[::-1][:n]
    out = []
    yy, xx = np.mgrid[0:h.shape[0], 0:h.shape[1]]
    for _, i in blobs:
        w = np.where(lab == i, h, 0)
        out.append(((w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()))
    if with_area:
        return np.array(out), [a for a, _ in blobs]
    return np.array(out)


def _zoom_rgb(img, s):
    """A sensor whose true scale is s * 0.1 mm/px sees everything 1/s times as large (about the centre)."""
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D(((W - 1) / 2, (H - 1) / 2), 0, 1.0 / s)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _calib_worker(args):
    method, s_true, trials, seed0, noisy, robot_sigma_mm = args
    from gelsight_sim import GelSightSim
    pm = _pm()
    sim = GelSightSim(mode="moving_sensor", seed=0, spec_edit=add_gauge)
    out = []
    for t in range(trials):
        rng = np.random.default_rng([seed0, t])
        sim.rng = np.random.default_rng([seed0, t, 1])
        ref = sim.reference_image()
        ref_z = _zoom_rgb(ref, s_true)

        def press_img(force, **pose):
            sim.press(force, **pose)
            img = _zoom_rgb(sim.tactile_image(), s_true)
            r2 = ref_z
            if noisy:
                img, r2 = structured_noise(img, ref_z, rng, None, drift=0.04, texture=2.0)
            return pm.height_map(img, r2, MM_PER_PX_CAM)[0]
        if method == "gauge":
            sim.place_object("calgauge", x=float(rng.uniform(-0.01, 0.01)), y=float(rng.uniform(-0.01, 0.01)),
                             yaw=float(rng.uniform(-np.pi, np.pi)))
            est = []
            for k in range(5):                             # 5 presses at small random offsets; median
                h = press_img(2.0, **sim.sensor_pose_for(x=float(rng.uniform(-0.002, 0.002)),
                                                          y=float(rng.uniform(-0.002, 0.002)),
                                                          yaw=float(rng.uniform(-np.pi, np.pi))))
                c, areas = _contact_centroids(h, 2, with_area=True)
                # the two contacts of a level press on two equal balls have about the same size
                if len(c) == 2 and min(areas) > 0.7 * max(areas):
                    est.append(GAUGE_SPACING_MM / np.linalg.norm(c[0] - c[1]))
            k_hat = float(np.median(est)) if est else np.nan
        else:                                              # robot displacement over a small ball
            sim.place_object("cal_ball", x=0.0, y=0.0)
            yaw = float(rng.uniform(-np.pi, np.pi))
            base = sim.sensor_pose_for(yaw=yaw)
            Sx = np.array([np.cos(yaw), np.sin(yaw)])
            Sy = np.array([np.sin(yaw), -np.cos(yaw)])     # sensor axes in the world (level sensor)
            D, P = [], []
            for dx in (-5.0, 0.0, 5.0):
                for dy in (-4.0, 0.0, 4.0):
                    cmd = np.array([dx, dy])               # commanded sensor offset in its own frame (mm)
                    move = (dx * Sx + dy * Sy) / 1000 + rng.normal(0, robot_sigma_mm / 1000, 2)
                    h = press_img(1.5, x=base["x"] + move[0], y=base["y"] + move[1], yaw=base["yaw"],
                                  pitch=base["pitch"], roll=base["roll"])
                    c = _contact_centroids(h, 1)
                    if len(c):
                        D.append(cmd)
                        P.append(c[0])
            D, P = np.array(D), np.array(P)
            A = np.c_[D, np.ones(len(D))]
            sol, *_ = np.linalg.lstsq(A, P, rcond=None)    # P = D @ B.T + c0  (moving the sensor moves the ball the other way)
            Bm = sol[:2].T
            k_hat = float(1.0 / np.sqrt(abs(np.linalg.det(Bm))))
        out.append(dict(method=method, s_true=s_true, noisy=noisy, k_hat=k_hat,
                        rel_err=float(k_hat / (MM_PER_PX_CAM * s_true) - 1)))
    return out


def calibrate(trials=40, workers=12):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    os.makedirs(V2, exist_ok=True)
    tasks = []
    for method in ("gauge", "displacement"):
        for s_true in (0.97, 1.0, 1.03):
            for noisy in (False, True):
                for part in range(2):
                    tasks.append((method, s_true, trials // 2, 1000 * part + int(s_true * 100) + 7 * noisy, noisy, 0.02))
    t0 = time.time()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        rows = sum(ex.map(_calib_worker, tasks), [])
    summ = {}
    for method in ("gauge", "displacement"):
        for noisy in (False, True):
            for s_true in (0.97, 1.0, 1.03):
                e = np.array([r["rel_err"] for r in rows if r["method"] == method and r["noisy"] == noisy
                              and r["s_true"] == s_true])
                e = e[np.isfinite(e)]
                key = f"{method}, {'structured noise' if noisy else 'clean'}, true scale {s_true}"
                summ[key] = dict(n=len(e), bias_pct=float(e.mean() * 100), std_pct=float(e.std() * 100),
                                 abs_p95_pct=float(np.percentile(np.abs(e), 95) * 100))
                print(f"{key:55s}: error {e.mean() * 100:+.3f} % +- {e.std() * 100:.3f} %, |err| p95 "
                      f"{np.percentile(np.abs(e), 95) * 100:.3f} % (n={len(e)})", flush=True)
    out = dict(procedures=dict(
        gauge=f"two {2 * GAUGE_BALL_R_MM:.0f} mm balls {GAUGE_SPACING_MM:.3f} mm apart; 5 presses at 2 N with random "
              "offsets/yaw; distance between the two contact centroids (weighted, reconstructed depth > 30 % of peak); "
              "presses whose two contacts differ in area by > 30 % are rejected; median over the presses",
        displacement="5 mm calibration ball; 3x3 grid of sensor offsets (+-5 x +-4 mm, robot repeatability sigma "
                     "0.02 mm); affine fit of contact centroid (px) vs commanded offset (mm)"),
        simulated_true_scales=[0.97, 1.0, 1.03], structured_noise="illumination drift 4 %, gel texture 2 grey levels",
        summary=summ, trials=rows, seconds=time.time() - t0)
    with open(os.path.join(V2, "calibration.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


# ================================================================ B: search loop v2
STEP_BACK_MM = 12.0


class SearcherV2(cs.Searcher):
    def __init__(self, perturb=None, model="new", threads=2):
        super().__init__(perturb, threads)
        if model == "new":
            import torch
            m = tc.build_model()
            m.load_state_dict(torch.load(os.path.join(V2, "model.pt"), map_location="cpu"))
            m.eval()
            self.model = m

    def episode_v2(self, i):
        """Static fork. Empty presses: the WHOLE gel footprint excludes the centre (kept in the belief);
        after an empty press, go back next to the last press with contact (backtrack_target) instead of
        trusting the empty area; after a contact press, the information-gain target as in v1."""
        st = self.st
        self.rng_noise = np.random.default_rng([int(i), 31])
        x0, y0, yaw, pitch, roll = st["arm_cmd"][i]
        force = float(st["force_cmd"][i])
        centre = np.array([x0, y0]) * 1000
        com = st["com_w"][i][:2] * 1000
        logp = np.zeros((cs.NW, cs.NW))
        rows = []
        x, y, pi, ro = x0, y0, pitch, roll
        last_contact = None
        for k in range(cs.MAX_PRESSES):
            X, o, R, info, touch = self.press(i, x, y, yaw, pi, ro, force, k)
            pa, pb, pc, H = cs.infer(self.model, X)
            contact = pa[2] >= 0.5
            if contact:
                Pw, _, _ = self.to_world(H, o, R, centre)
                logp += np.log((1 - cs.EPS_MIX) * Pw + cs.EPS_MIX / (cs.NW * cs.NW))
                last_contact = np.array([x, y]) * 1000
            else:
                _, sx, sy = self.to_world(H, o, R, centre)
                inside = (np.abs(sx) <= tc.PAD_HALF_MM[0]) & (np.abs(sy) <= tc.PAD_HALF_MM[1])   # whole gel footprint
                logp += np.where(inside, np.log(cs.EPS_EMPTY), 0.0)
            P = np.exp(logp - logp.max())
            P /= P.sum()
            area, thr = self.region90(P)
            am = np.unravel_index(P.argmax(), P.shape)
            est = centre + np.array([self.gx[am], self.gy[am]])
            ic = np.clip(((com - centre + cs.WORLD_HALF) / cs.WORLD_CELL).astype(int), 0, cs.NW - 1)
            rows.append(dict(press=k + 1, x_mm=float(x * 1000), y_mm=float(y * 1000), p_fork=float(pa[2]),
                             fork_touch=bool(touch), err_mm=float(np.linalg.norm(est - com)), area90_mm2=float(area),
                             truth_in_90=bool(P[ic[1], ic[0]] >= thr), housing=bool(info.housing_contact),
                             est_mm=est.tolist(), backtrack=False))
            if area < cs.STOP_AREA_MM2:
                break
            pi = ro = 0.0
            if contact or last_contact is None:
                nxt = self.info_target(P, centre, yaw)
            else:
                nxt = self.backtrack_target(P, centre, yaw, last_contact, [(r["x_mm"], r["y_mm"]) for r in rows])
                rows[-1]["backtrack"] = True
            x, y = float(np.clip(nxt[0] / 1000, -0.148, 0.148)), float(np.clip(nxt[1] / 1000, -0.148, 0.148))
        self.post = P
        return dict(index=int(i), strategy="v2", presses=len(rows), converged=rows[-1]["area90_mm2"] < cs.STOP_AREA_MM2,
                    final_err_mm=rows[-1]["err_mm"], final_truth_in_90=rows[-1]["truth_in_90"],
                    first_err_mm=rows[0]["err_mm"], trace=rows, com_mm=com.tolist(), centre_mm=centre.tolist(),
                    yaw=float(yaw), pose=int(st["pose_id"][i]), would_move=bool(st["would_move"][i]))


    def backtrack_target(self, P, centre, yaw, last_contact, visited):
        """After an empty press: stay next to the last press that touched the fork. Candidates on rings of
        STEP_BACK_MM (then 2x) around it, 8 directions; skip spots already pressed (< 4 mm); take the one
        whose gel footprint holds the most belief (empty footprints were already down-weighted in P)."""
        c_, s_ = np.cos(yaw), np.sin(yaw)
        ex, ey = np.array([c_, s_]), np.array([s_, -c_])
        WX = (centre[0] + self.gx).ravel()
        WY = (centre[1] + self.gy).ravel()
        Pf = P.ravel()
        vis = np.array(visited)
        for ring in (STEP_BACK_MM, 2 * STEP_BACK_MM):
            best, best_m = None, -1.0
            for k in range(8):
                th = k * np.pi / 4
                cand = last_contact + ring * np.array([np.cos(th), np.sin(th)])
                if len(vis) and np.min(np.linalg.norm(vis - cand, axis=1)) < 4.0:
                    continue
                dx, dy = WX - cand[0], WY - cand[1]
                inside = ((np.abs(dx * ex[0] + dy * ex[1]) <= tc.PAD_HALF_MM[0])
                          & (np.abs(dx * ey[0] + dy * ey[1]) <= tc.PAD_HALF_MM[1]))
                m = float(Pf[inside].sum())
                if m > best_m:
                    best, best_m = cand, m
            if best is not None:
                return best
        return np.asarray(last_contact, float)


def _search_worker_v2(args):
    idx, model, loop, perturb = args
    S = SearcherV2(perturb, model)
    return [S.episode_v2(i) if loop == "v2" else S.episode(i, "info") for i in idx]


def search_v2(models=("new",), loops=("v2",), conditions=("clean",), n=300, workers=16):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context
    out_dir = os.path.join(V2, "search")
    os.makedirs(out_dir, exist_ok=True)
    C = tc.load_cache()
    # the same 300 first presses as the v1 search (random test presses on the fork, natural poses)
    test = np.nonzero((C["split"] == 2) & C["natural"] & (C["sample_type"] == 0) & (C["cls"] == 2))[0]
    pick = np.sort(np.random.default_rng(2026).choice(test, min(n, len(test)), replace=False))
    idx = C["index"][pick]
    print(f"rendering on EGL device {os.environ.get('MUJOCO_EGL_DEVICE_ID', 'default')}", flush=True)
    t0 = time.time()
    for model in models:
        for loop in loops:
            for cond in conditions:
                chunks = [list(idx[k::workers]) for k in range(workers)]
                with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
                    eps = sum(ex.map(_search_worker_v2, [(c, model, loop, cs.PERTURB[cond]) for c in chunks]), [])
                eps.sort(key=lambda e: e["index"])
                for e in eps:
                    g = pick[np.nonzero(idx == e["index"])[0][0]]
                    e["first_part"] = tc.PARTS4[C["part"][g]]
                name = f"{model}_{loop}_{cond}"
                with open(os.path.join(out_dir, f"episodes_{name}.json"), "w") as f:
                    json.dump(eps, f)
                print(f"{name:28s} " + search_line(eps) + f"  ({time.time() - t0:.0f} s)", flush=True)


def thresholds(e, area):
    for r in e["trace"]:
        if r["area90_mm2"] < area:
            return r
    return None


def search_stats(eps):
    out = dict(n=len(eps))
    for area in (16.0, 4.0):
        hits = [thresholds(e, area) for e in eps]
        reached = np.array([h is not None for h in hits])
        ok = np.array([h is not None and h["truth_in_90"] for h in hits])
        errs = np.array([h["err_mm"] for h in hits if h is not None])
        presses = np.array([h["press"] for h in hits if h is not None])
        out[f"<{area:.0f}mm2"] = dict(reached=float(reached.mean()), success=float(ok.mean()),
                                      presses_median=float(np.median(presses)) if len(presses) else None,
                                      err_mm_median=float(np.median(errs)) if len(errs) else None,
                                      err_mm_p90=float(np.percentile(errs, 90)) if len(errs) else None,
                                      err_gt5mm_when_reached=float(np.mean(errs > 5)) if len(errs) else None)
    fe = np.array([e["final_err_mm"] for e in eps])
    out["final"] = dict(err_mm_median=float(np.median(fe)), err_mm_p90=float(np.percentile(fe, 90)),
                        err_gt5mm=float(np.mean(fe > 5)), truth_in_90=float(np.mean([e["final_truth_in_90"] for e in eps])),
                        presses_mean=float(np.mean([e["presses"] for e in eps])))
    return out


def search_line(eps):
    s = search_stats(eps)
    a, b, f = s["<16mm2"], s["<4mm2"], s["final"]
    e16 = "-" if a["err_mm_median"] is None else f"{a['err_mm_median']:.1f}"     # None: never reached
    return (f"<16 mm2: success {a['success'] * 100:.0f} % (reached {a['reached'] * 100:.0f} %, {a['presses_median']} presses, "
            f"err {e16} mm) | <4 mm2: success {b['success'] * 100:.0f} % (reached {b['reached'] * 100:.0f} %, "
            f"{b['presses_median']} presses) | final err {f['err_mm_median']:.2f}/{f['err_mm_p90']:.1f} mm, "
            f">5 mm {f['err_gt5mm'] * 100:.1f} %")


def search_report_v2():
    import glob
    out_dir = os.path.join(V2, "search")
    runs = {}
    for fn in sorted(glob.glob(os.path.join(out_dir, "episodes_*.json"))):
        runs[os.path.basename(fn)[9:-5]] = json.load(open(fn))
    for cond in ("clean", "noise_0.05mm", "lateral_0.98"):              # the previous loop with the old model (v1 search)
        fn = os.path.join(cs.SEARCH, f"episodes_{cond}.json")
        if os.path.exists(fn):
            runs[f"old_v1_{cond} (previous search)"] = json.load(open(fn))[f"{cond}/info"]
    rep = {}
    for name, eps in runs.items():
        r = dict(all=search_stats(eps))
        for lab, sel in [("first press on raised tines (would_move)", lambda e: e["would_move"]),
                         ("first press elsewhere", lambda e: not e["would_move"])] + \
                        [(f"first press on {p}", (lambda e, p=p: e["first_part"] == p)) for p in tc.PARTS4]:
            sub = [e for e in eps if sel(e)]
            if sub:
                r[lab] = search_stats(sub)
        r["failures"] = [dict(index=e["index"], first_part=e["first_part"], pose=e["pose"], would_move=e["would_move"],
                              presses=e["presses"], final_err_mm=round(e["final_err_mm"], 1))
                         for e in eps if e["final_err_mm"] > 5]
        r["backtracks_per_episode"] = float(np.mean([sum(t.get("backtrack", False) for t in e["trace"]) for e in eps]))
        rep[name] = r
        print(f"{name:40s} " + search_line(eps))
    with open(os.path.join(out_dir, "search_metrics.json"), "w") as f:
        json.dump(rep, f, indent=1)
    return rep, runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["bank", "train", "evaluate", "tolerance", "calibrate", "search", "search-report"])
    ap.add_argument("--include-edge-poses", action="store_true",
                    help="train: also train on the propped (side-edge) poses (no pose label for them)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=["new"])
    ap.add_argument("--loops", nargs="+", default=["v2"])
    ap.add_argument("--conditions", nargs="+", default=["clean", "noise_0.05mm", "lateral_0.98"])
    args = ap.parse_args()
    if tc.SENSOR != "standard":
        raise SystemExit(f"centre_v2.py works on dataset/ (standard sensor); GELSIGHT_SENSOR={tc.SENSOR}: use centre_v3.py")
    if args.cmd == "bank":
        build_bank()
    elif args.cmd == "train":
        train_v2(args.include_edge_poses, args.epochs, args.seed,
                 name="model.pt" if not args.include_edge_poses else "model_edge.pt")
    elif args.cmd == "evaluate":
        evaluate_v2()
    elif args.cmd == "tolerance":
        tolerance()
    elif args.cmd == "calibrate":
        calibrate()
    elif args.cmd == "search":
        search_v2(args.models, args.loops, args.conditions)
        search_report_v2()
    elif args.cmd == "search-report":
        search_report_v2()


if __name__ == "__main__":
    main()
