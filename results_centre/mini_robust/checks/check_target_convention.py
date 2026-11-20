"""Does augment_robust move the target the way the network sees the image move? An existing Mini model (trained
without shifts) on test2 fork presses, transformed by augment_robust with fixed draws: 90 degree turn, 2 px shift.
Its prediction should follow the new target (vr), not the alternatives."""
import os, sys
os.environ.setdefault("GELSIGHT_SENSOR", "mini")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np, torch
import train_centre as tc, centre_v2 as v2, centre_v3 as v3
dev = "cuda"
C = v3.load3()
idx = np.nonzero((C["split"] == 4) & (C["cls"] == 2))[0][:1500]
net = v3.load_net(os.path.join(v3.HERE, "results_centre", "mini_exact", "model_100k_seed0.pt"), dev)
x = torch.tensor(C["X"][idx].astype(np.float32), device=dev)[:, None]
vec = torch.tensor(C["vec_mm"][idx].astype(np.float32), device=dev)
gen = torch.Generator(device=dev).manual_seed(0)
coords = tc.coord_channels(dev)


def pred(xi):
    am = []
    with torch.no_grad():
        for i in range(0, len(xi), 128):
            xb = xi[i:i + 128]
            am.append(net(torch.cat([xb, coords.expand(len(xb), -1, -1, -1)], 1))[3].float().argmax(1).cpu().numpy())
    am = np.concatenate(am)
    return np.stack([tc.CELL_C[am % tc.NC], tc.CELL_C[am // tc.NC]], 1)


base = dict(blur=9, noise=0.0, test_border=False, treat=0, untouched=False, sc=1.0, kern=1, mir=1.0)
p0 = pred(x)
e0 = np.linalg.norm(p0 - vec.cpu().numpy(), axis=1)
print(f"as rendered: median error {np.median(e0):.3f} mm")
for name, P in (("turn +90 deg", dict(base, th=np.pi / 2, shift=0.0)), ("turn -90 deg", dict(base, th=-np.pi / 2, shift=0.0)),
                ("shift +2 px in x", dict(base, th=0.0, shift=torch.tensor([2.0, 0.0], device=dev).expand(len(idx), 2))),
                ("shift +2 px in y", dict(base, th=0.0, shift=torch.tensor([0.0, 2.0], device=dev).expand(len(idx), 2)))):
    xo, vr, d = v2.augment_robust(x, vec, gen, params=P)
    p = pred(xo)
    vr = vr.cpu().numpy()
    e = np.linalg.norm(p - vr, axis=1)
    msg = f"{name:17s}: median error vs the new target {np.median(e):.3f} mm"
    if "turn" in name:
        alt = (np.array([[0, 1], [-1, 0]]) if "+90" in name else np.array([[0, -1], [1, 0]])) @ vec.cpu().numpy().T
        msg += f" | vs the opposite turn {np.median(np.linalg.norm(p - alt.T, axis=1)):.2f} mm"
    else:
        dm = (p - p0).mean(0)
        msg += f" | mean predicted move ({dm[0]:+.3f}, {dm[1]:+.3f}) mm, target move ({(vr - vec.cpu().numpy()).mean(0)[0]:+.3f}, {(vr - vec.cpu().numpy()).mean(0)[1]:+.3f}) mm"
    print(msg)
