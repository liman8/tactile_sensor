"""TTA wrapper checks on CPU: identity-only TTA reproduces the plain network; the full set keeps the error."""
import os, sys
os.environ.setdefault("GELSIGHT_SENSOR", "mini")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np, torch
import train_centre as tc, centre_v3 as v3
torch.set_num_threads(8)
C = v3.load3()
idx = np.nonzero((C["split"] == 4) & (C["cls"] == 2))[0][:300]
net = v3.load_net(os.path.join(v3.HERE, "results_centre", "mini_exact", "model_100k_seed0.pt"), "cpu")
pa0, pb0, pc0, P0 = tc.predict(net, C, idx, bs=64)
ident = v3.tta_wrap(net, [["identity", 0, 0, 1.0]]).eval()
pa1, pb1, pc1, P1 = tc.predict(ident, C, idx, bs=64)
print(f"identity TTA vs plain: max |dP| heatmap {np.abs(P1 - P0).max():.2e}, heads {max(np.abs(pa1 - pa0).max(), np.abs(pb1 - pb0).max()):.2e}")
full = v3.tta_wrap(net, v3.TTA_DEFAULT).eval()
pa2, pb2, pc2, P2 = tc.predict(full, C, idx, bs=64)
for name, P in (("plain", P0), ("full TTA", P2)):
    hm = tc.heat_metrics(P, C["vec_mm"][idx])
    print(f"{name:9s}: median {np.median(hm['argmax_err_mm']):.3f} mm, p90 {np.percentile(hm['argmax_err_mm'], 90):.2f} mm, "
          f"centre in 90 % {hm['cover_90'].mean() * 100:.1f} %, probabilities sum {P.sum(1).mean():.4f}")
# mean predicted position: the shift pairs must not move it
def mean_pt(P):
    am = P.argmax(1)
    return np.stack([tc.CELL_C[am % tc.NC], tc.CELL_C[am // tc.NC]], 1)
print("mean argmax shift full TTA - plain (mm):", np.round((mean_pt(P2) - mean_pt(P0)).mean(0), 3))
