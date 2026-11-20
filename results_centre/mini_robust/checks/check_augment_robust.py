"""Checks of centre_v2.augment_robust on synthetic Mini inputs (GELSIGHT_SENSOR=mini)."""
import os, sys, time
os.environ.setdefault("GELSIGHT_SENSOR", "mini")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np, torch
import train_centre as tc, centre_v2 as v2
dev = "cuda"
H, W, mm = tc.H_IN, tc.W_IN, tc.MM_PER_PX_IN
gen = torch.Generator(device=dev).manual_seed(5)
clean = dict(blur=9, noise=0.0, test_border=False, treat=0)          # no blur / noise / border
yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32), torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
cx, cy = (W - 1) / 2, (H - 1) / 2
ok = True

# 1. untouched = exact permutation (flips) of the input
B = 512
x = torch.rand(B, 1, H, W, device=dev)
vec = torch.zeros(B, 2, device=dev)
out, vr, d = v2.augment_robust(x, vec, gen, params=dict(clean, untouched=True))
ref = x.clone()
ref = torch.where((d["mir"] < 0)[:, None, None, None], torch.flip(ref, [3]), ref)
ref = torch.where((torch.cos(d["th"]) < 0)[:, None, None, None], torch.flip(ref, [2, 3]), ref)
e = (out - ref).abs().max().item()
print(f"1. untouched: max |out - flipped input| = {e:.2e}; every value is an input value: "
      f"{bool(torch.isin(out, x).all())}"); ok &= e == 0

# 2. image and target move together (zoom 1): a blob at the target, then its centroid vs the new target
B = 2000
vec = torch.stack([torch.empty(B, device=dev).uniform_(-5, 5), torch.empty(B, device=dev).uniform_(-4, 4)], 1)
bx, by = cx + vec[:, 0] / mm, cy + vec[:, 1] / mm
x = 10 * torch.exp(-((xx[None] - bx[:, None, None]) ** 2 + (yy[None] - by[:, None, None]) ** 2) / (2 * 2.0 ** 2))[:, None]
for zoom in (1.0, 1.03):
    out, vr, d = v2.augment_robust(x, vec, gen, params=dict(clean, untouched=False, sc=zoom, kern=1))
    o = out[:, 0]
    cen = torch.stack([(o * xx).sum((1, 2)) / o.sum((1, 2)), (o * yy).sum((1, 2)) / o.sum((1, 2))], 1)
    cen_mm = torch.stack([(cen[:, 0] - cx) * mm, (cen[:, 1] - cy) * mm], 1)
    expect = zoom * (vr - d["shift"] * mm) + d["shift"] * mm         # image zoomed, target not; shift on both
    dev_t = (cen_mm - vr).norm(dim=1)
    dev_e = (cen_mm - expect).norm(dim=1)
    print(f"2. zoom {zoom}: blob centroid vs new target: median {dev_t.median().item():.4f} mm, max {dev_t.max().item():.4f} mm; "
          f"vs zoom-corrected expectation: median {dev_e.median().item():.4f}, max {dev_e.max().item():.4f} mm (shift drawn 0-1 px: "
          f"{d['shift'].min().item():.2f}-{d['shift'].max().item():.2f})")
    ok &= dev_e.max().item() < 0.05      # bilinear zoom of a sampled blob: centroid within ~0.1 px

# 3. kernels: constant image stays 1 in the interior; nearest only takes input values
x = torch.ones(256, 1, H, W, device=dev)
for k, name in enumerate(v2.ROBUST_KERNELS):
    out, _, d = v2.augment_robust(x, torch.zeros(256, 2, device=dev), gen, params=dict(clean, untouched=False, kern=k, th=0.0, mir=1.0))
    inner = out[:, 0, 4:-4, 4:-4]
    print(f"3. kernel {name:8s}: interior of a constant image min {inner.min().item():.4f} max {inner.max().item():.4f}")
xr = torch.rand(64, 1, H, W, device=dev)
out, _, _ = v2.augment_robust(xr, torch.zeros(64, 2, device=dev), gen, params=dict(clean, untouched=False, kern=0))
print(f"3. nearest: every output value is an input value or 0: {bool(torch.isin(out, torch.cat([xr.flatten(), torch.zeros(1, device=dev)])).all())}")

# 4. border treatments on a gradient image
x = (yy * 100 + xx)[None, None].expand(5 * 4, 1, H, W).clone()        # value encodes (row, col)
x = x + 1000
treat = torch.arange(5, device=dev).repeat_interleave(4)
width = torch.arange(1, 5, device=dev).repeat(5)
P = dict(blur=9, noise=0.0, test_border=False, untouched=True, th=0.0, mir=1.0,
         treat=treat[:, None].expand(20, 4), width=width[:, None].expand(20, 4), fac=torch.full((20, 4), 0.5, device=dev))
out, _, _ = v2.augment_robust(x, torch.zeros(20, 2, device=dev), gen, params=P)
for i in range(20):
    t, w = v2.BORDER_TREATMENTS[treat[i]], int(width[i])
    o, xi = out[i, 0], x[i, 0]
    band = torch.zeros(H, W, dtype=torch.bool, device=dev); band[:w] = band[-w:] = True; band[:, :w] = band[:, -w:] = True
    inner_same = bool((o[~band] == xi[~band]).all())
    if t == "none": good = bool((o == xi).all())
    elif t == "mask": good = bool((o[band] == 0).all())
    elif t in ("darken", "brighten"): good = True        # checked below
    else:   # replicate: outer rows = row w, outer cols = col w (corners both)
        good = bool((o[0, w:-w] == xi[w, w:-w]).all() and (o[w:-w, 0] == xi[w:-w, w]).all() and o[0, 0] == xi[w, w])
    if t in ("darken", "brighten"):
        f = 0.5 if t == "darken" else 1.25
        edge_only = band.clone(); edge_only[:w, :w] = edge_only[:w, -w:] = edge_only[-w:, :w] = edge_only[-w:, -w:] = False
        corner = band & ~edge_only
        good = bool(torch.allclose(o[edge_only], xi[edge_only] * f) and torch.allclose(o[corner], xi[corner] * f * f))
    if i % 4 == 0 or not (good and inner_same):
        print(f"4. border {t:9s} width {w}: band as specified {good}, inside untouched {inner_same}")
    ok &= good and inner_same

# 5. how often each draw happens (defaults)
B = 20000
x = torch.rand(B, 1, H, W, device=dev)
out, vr, d = v2.augment_robust(x, torch.zeros(B, 2, device=dev), gen)
res = ~d["untouched"]
print(f"5. untouched {d['untouched'].float().mean().item() * 100:.1f} %, test border {d['test_border'].float().mean().item() * 100:.1f} %, "
      f"kernels among resampled " + ", ".join(f"{n} {(d['kern'][res] == k).float().mean().item() * 100:.1f} %" for k, n in enumerate(v2.ROBUST_KERNELS)))
nt = ~d["test_border"]
print("   border treatments per side (non-test samples): " + ", ".join(f"{n} {(d['treat'][nt] == k).float().mean().item() * 100:.1f} %" for k, n in enumerate(v2.BORDER_TREATMENTS))
      + f"; widths 1-4: " + " ".join(f"{(d['width'][nt] == w).float().mean().item() * 100:.0f}%" for w in range(1, 5)))
print(f"   zoom among resampled {d['sc'][res].min().item():.3f}-{d['sc'][res].max().item():.3f}, mean {d['sc'][res].mean().item():.4f}; "
      f"shift {d['shift'][res].min().item():.3f}-{d['shift'][res].max().item():.3f} px")

# 6. speed, batch 64
xb = torch.rand(64, 1, H, W, device=dev); vb = torch.zeros(64, 2, device=dev)
for f, name in ((lambda: v2.augment_v2(xb, vb, gen), "augment_v2"), (lambda: v2.augment_robust(xb, vb, gen), "augment_robust")):
    for _ in range(5): f()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(200): f()
    torch.cuda.synchronize(); print(f"6. {name}: {(time.time() - t0) / 200 * 1000:.2f} ms per batch of 64")
print("ALL CHECKS PASSED" if ok else "SOME CHECK FAILED")
