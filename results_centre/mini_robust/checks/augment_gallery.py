"""A few real Mini training inputs through augment_robust (and the test-time border mask)."""
import os, sys
os.environ.setdefault("GELSIGHT_SENSOR", "mini")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import train_centre as tc, centre_v2 as v2, centre_v3 as v3
C = v3.load3()
idx = np.nonzero((C["split"] == 4) & (C["cls"] == 2))[0][[3, 40]]
gen = torch.Generator(device="cuda").manual_seed(3)
fig, ax = plt.subplots(2, 7, figsize=(21, 5.2))
for r, i in enumerate(idx):
    x = torch.tensor(C["X"][[i] * 6].astype(np.float32), device="cuda")[:, None]
    out, vr, d = v2.augment_robust(x, torch.tensor(C["vec_mm"][[i] * 6].astype(np.float32), device="cuda"), gen)
    ims = [x[0, 0].cpu().numpy()] + [o[0].cpu().numpy() for o in out]
    tt = ["as rendered"] + [f"{'untouched' if d['untouched'][k] else v2.ROBUST_KERNELS[d['kern'][k]] + f' z {d[chr(115) + chr(99)][k]:.3f}'}\n"
                            + ("test border" if d["test_border"][k] else "/".join(v2.BORDER_TREATMENTS[t][:3] + str(int(w))
                               for t, w in zip(d["treat"][k].tolist(), d["width"][k].tolist()))) for k in range(6)]
    vmax = ims[0].max()
    for c, (im, t) in enumerate(zip(ims, tt)):
        ax[r, c].imshow(im, cmap="magma", vmin=-0.05, vmax=vmax)
        ax[r, c].set_title(t, fontsize=8)
        ax[r, c].axis("off")
fig.suptitle("augment_robust on two test presses (depth, mm). Border: top/bottom/left/right treatment + width in px", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "augment_gallery.png"), dpi=80)
