"""
Training-free baseline: analysis-by-synthesis localisation of the fork's centre with its CAD.

For every natural resting pose the fork's top-surface height map is rendered once, for yaws
0..358 deg in 2 deg steps, in the sensor-image convention (x = image column, y = image row,
height up). A press's reconstructed depth map is then matched against them:

  1. coarse (0.8 mm/px): normalised cross-correlation over every placement of the 32 x 24 mm
     window on every rotated template, by FFT (2 poses x 180 yaws in one batch);
  2. the coarse NCC volume, turned into weights exp((NCC - max) / tau1), is mapped to the fork's
     centre of mass relative to the sensor (each placement fixes where the centre is) -> a coarse
     probability map;
  3. refinement (0.4 mm/px) of the top-K placements (after non-maximum suppression): +-1 px, +-2 deg,
     and for each the press model  depth ~ s * blur(max(0, height - z0)) + b  with the best z0
     (12 levels), s and b in closed form; candidate weight exp(-(e - e_min) / tau2), e = unexplained
     fraction of the depth variance;
  4. final map = (1 - lam) * refined candidates (splatted, sigma 1.5 mm) + lam * coarse map, on the
     same 150 x 150 grid (+-150 mm, 2 mm cells) as the network's heatmap.

Contact: the press counts as "fork" when the smoothed depth has a peak > 0.08 mm over > 1 mm2 and
the best NCC exceeds eta. tau1, tau2, lam and eta are the only fitted numbers (calibrate on
validation presses with centre_v3.py).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(HERE, "dataset_v3", "abs_templates.npz")
RES = 0.4                      # mm per px of the stored templates (refinement)
HALF_CANVAS_MM = 82.0          # fork radius about its bbox centre (~62 mm) + window half-diagonal (20 mm)
YAW_STEP_DEG = 2.0
NAT_POSES = (0, 1)
GRID_HALF, GRID_CELL, NG = 150.0, 2.0, 150


def build_templates(path=TEMPLATES):
    """Top-surface height (mm above the table) of the fork for each natural pose and yaw, in the
    sensor-image convention: sensor x = pose x, sensor y = -pose y (the sensor looks down), then
    rotated by yaw about the sensor z axis. Also the centre of mass position per template."""
    import trimesh
    from collect_dataset import rasterize
    import train_centre as tc
    mesh = trimesh.load(os.path.join(HERE, "meshes", "fork_stl", "fork_stl_visual.stl"), process=True)
    poses = {p["id"]: p for p in json.load(open(os.path.join(HERE, "dataset", "poses", "stable_poses.json")))["poses"]}
    n = int(round(2 * HALF_CANVAS_MM / RES))
    yaws = np.arange(0, 360, YAW_STEP_DEG)
    T = np.zeros((len(NAT_POSES), len(yaws), n, n), np.float16)
    com = np.zeros((len(NAT_POSES), len(yaws), 2), np.float32)
    t0 = time.time()
    for a, pid in enumerate(NAT_POSES):
        p = poses[pid]
        R, rest = tc.quat2mat(p["quat_wxyz"]), np.array(p["rest_pos_m"]) * 1000
        V = mesh.vertices @ R.T + rest                         # pose frame, mm; bbox centre at the origin
        c = R @ mesh.center_mass + rest
        s0 = np.c_[V[:, 0], -V[:, 1]]
        c0 = np.array([c[0], -c[1]])
        for b, yaw in enumerate(yaws):
            th = np.radians(yaw)
            Rt = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            s = s0 @ Rt.T
            P = np.c_[s[:, 0] / RES + (n - 1) / 2, s[:, 1] / RES + (n - 1) / 2, V[:, 2]]
            z, tid = rasterize(P, mesh.faces, n, n, keep="max")
            T[a, b] = np.where(tid >= 0, np.maximum(z, 0), 0).astype(np.float16)
            com[a, b] = Rt @ c0
        print(f"  templates pose {pid}: {len(yaws)} yaws ({time.time() - t0:.0f} s)", flush=True)
    np.savez_compressed(path, T=T, com=com, yaws=yaws, res=RES, half=HALF_CANVAS_MM, poses=np.array(NAT_POSES))
    return path


class ABSLocalizer:
    DEFAULT = dict(tau1=0.02, tau2=0.02, lam=0.05, eta=0.3, K=48, n_top=800, blur_mm=0.45)

    def __init__(self, device="cpu", params=None, path=TEMPLATES):
        import torch
        self.torch = torch
        self.dev = torch.device(device)
        d = np.load(path)
        T = torch.tensor(d["T"])                                     # (P, Y, n, n) at 0.4 mm, float16
        self.P, self.Y, self.n = T.shape[0], T.shape[1], T.shape[2]
        self.Tf = T.reshape(self.P * self.Y, self.n, self.n).to(self.dev)          # kept in float16 (memory)
        self.Tc = torch.nn.functional.avg_pool2d(self.Tf[:, None].float(), 2)[:, 0]   # 0.8 mm, float32
        self.com = torch.tensor(d["com"].reshape(-1, 2), device=self.dev)     # mm, sensor convention
        self.yaws = d["yaws"]
        self.p = dict(self.DEFAULT, **(params or {}))
        nc = self.Tc.shape[-1]
        self.FT = None
        self._fft_shape = (nc, nc)
        self.FT = torch.fft.rfft2(self.Tc, s=self._fft_shape)
        # integral images of T and T^2 (coarse) for the window statistics
        self.I1 = self._integral(self.Tc)
        self.I2 = self._integral(self.Tc ** 2)
        g = self.p["blur_mm"] / RES
        r = int(np.ceil(3 * g))
        k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * g * g))
        self.gk = (k / k.sum()).to(self.dev)

    @staticmethod
    def _integral(x):
        import torch
        return torch.nn.functional.pad(x.cumsum(-1).cumsum(-2), (1, 0, 1, 0))

    def _box(self, I, h, w):
        return I[:, h:, w:] - I[:, :-h, w:] - I[:, h:, :-w] + I[:, :-h, :-w]

    def _blur(self, x):
        import torch.nn.functional as F
        k = self.gk
        r = (len(k) - 1) // 2
        y = F.conv2d(F.pad(x[:, None], (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
        y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
        return y[:, 0]

    def predict(self, X):
        """X: reconstructed depth (120, 160) mm at 0.2 mm/px. Returns (p_fork, heatmap (150, 150), info)."""
        raw = self.predict_raw(X, (self.p["tau1"],))
        return self.compose(raw, self.p["tau1"], self.p["tau2"], self.p["lam"], self.p["eta"])

    def compose(self, raw, tau1, tau2, lam, eta):
        """Probability map from the raw matching result (cheap: used to fit tau1, tau2, lam, eta)."""
        uni = np.full((NG, NG), 1.0 / (NG * NG), np.float32)
        if not raw["contact"] or raw["ncc"] < eta:
            return 0.0, uni, raw["info"]
        ev = raw["e"]
        wv = np.exp(-(ev - ev.min()) / tau2)
        ref = np.zeros((NG, NG), np.float64)
        for (vxk, vyk), wk in zip(raw["v"], wv):
            gx = (vxk + GRID_HALF) / GRID_CELL - 0.5
            gy = (vyk + GRID_HALF) / GRID_CELL - 0.5
            x0, y0 = int(np.floor(gx)), int(np.floor(gy))
            xs_ = np.arange(x0 - 2, x0 + 4)
            ys_ = np.arange(y0 - 2, y0 + 4)
            xs_, ys_ = xs_[(xs_ >= 0) & (xs_ < NG)], ys_[(ys_ >= 0) & (ys_ < NG)]
            if len(xs_) and len(ys_):
                dd = ((xs_[None] - gx) ** 2 + (ys_[:, None] - gy) ** 2) * GRID_CELL ** 2
                ref[np.ix_(ys_, xs_)] += wk * np.exp(-dd / (2 * 1.5 ** 2))
        ref /= ref.sum() + 1e-12
        Hm = (1 - lam) * ref + lam * raw["cmaps"][tau1]
        return 1.0, (Hm / Hm.sum()).astype(np.float32), raw["info"]

    def predict_raw(self, X, tau1s=(0.01, 0.02, 0.04, 0.08)):
        """Matching without the fitted constants: coarse maps for several tau1, refined candidates (e, centre)."""
        torch = self.torch
        F = torch.nn.functional
        x = torch.tensor(np.asarray(X, np.float32), device=self.dev)
        xs = self._blur(F.avg_pool2d(x[None, None], 2)[:, 0])[0]           # 0.4 mm, (60, 80), lightly smoothed
        a2 = (0.4 ** 2)
        peak = float(xs.max())
        area = float((xs > 0.05).sum()) * a2
        if peak < 0.08 or area < 1.0:
            return dict(contact=False, ncc=-1.0, info=dict(peak=peak, area=area, ncc=np.nan))
        # ---- coarse NCC by FFT
        xc = F.avg_pool2d(xs[None, None], 2)[0, 0]                           # 0.8 mm, (30, 40)
        h, w = xc.shape
        xz = xc - xc.mean()
        nx = torch.sqrt((xz ** 2).sum()) + 1e-9
        FX = torch.fft.rfft2(xz, s=self._fft_shape)
        num = torch.fft.irfft2(self.FT * FX.conj(), s=self._fft_shape)
        nc = self._fft_shape[0]
        num = num[:, :nc - h + 1, :nc - w + 1]
        s1 = self._box(self.I1, h, w)
        s2 = self._box(self.I2, h, w)
        var = torch.clamp(s2 - s1 ** 2 / (h * w), min=0)
        ncc = num / (nx * torch.sqrt(var + 0.0025 * h * w))                # floor: template std >= 0.05 mm
        best = float(ncc.max())
        # ---- coarse map over the centre of mass
        Nt, R_, C_ = ncc.shape
        rr = torch.arange(R_, device=self.dev, dtype=torch.float32)
        cc = torch.arange(C_, device=self.dev, dtype=torch.float32)
        cres = 2 * RES
        ctr = (nc - 1) / 2
        vx = (ctr + self.com[:, 0, None] / cres - (cc[None] + (w - 1) / 2)) * cres        # (Nt, C)
        vy = (ctr + self.com[:, 1, None] / cres - (rr[None] + (h - 1) / 2)) * cres        # (Nt, R)
        ix = torch.clamp(((vx + GRID_HALF) / GRID_CELL).long(), 0, NG - 1)
        iy = torch.clamp(((vy + GRID_HALF) / GRID_CELL).long(), 0, NG - 1)
        flat = (iy[:, :, None] * NG + ix[:, None, :]).reshape(-1)
        cmaps = {}
        for t1 in tau1s:
            wgt = torch.exp((ncc - best) / t1)
            cm_ = torch.zeros(NG * NG, device=self.dev).index_add_(0, flat, wgt.reshape(-1))
            cmaps[t1] = (cm_ / cm_.sum()).reshape(NG, NG).cpu().numpy()
        # ---- top-K placements with non-maximum suppression, refined at 0.4 mm
        K = self.p["K"]
        flat_ncc = ncc.reshape(-1)
        top = torch.topk(flat_ncc, min(int(self.p["n_top"]), flat_ncc.numel())).indices
        cand = []
        for t in top.tolist():
            k, rem = divmod(t, R_ * C_)
            r, c = divmod(rem, C_)
            ok = True
            for (k2, r2, c2) in cand:
                if k2 // self.Y == k // self.Y and min(abs(k2 - k), self.Y - abs(k2 - k)) <= 2 and abs(r2 - r) <= 3 and abs(c2 - c) <= 3:
                    ok = False
                    break
            if ok:
                cand.append((k, r, c))
            if len(cand) >= K:
                break
        hf, wf = xs.shape
        n = self.n
        win, meta = [], []
        for ci, (k, r, c) in enumerate(cand):
            pose, yi = divmod(k, self.Y)
            for dy in (-1, 0, 1):
                kk = pose * self.Y + (yi + dy) % self.Y
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        r0, c0 = 2 * r + dr, 2 * c + dc
                        if 0 <= r0 and 0 <= c0 and r0 + hf <= n and c0 + wf <= n:
                            win.append(self.Tf[kk, r0:r0 + hf, c0:c0 + wf])
                            meta.append((ci, kk, r0, c0))
        Wt = torch.stack(win).float()                                       # (M, 60, 80)
        top_h = Wt.reshape(len(Wt), -1).max(1).values
        xv = xs.reshape(-1)
        xm = xv - xv.mean()
        vx_tot = (xm ** 2).sum() + 1e-9
        best_e = torch.full((len(Wt),), float("inf"), device=self.dev)
        for d in np.linspace(0.15, 3.0, 12):                                 # press depth into the top surface
            pred = self._blur(torch.clamp(Wt - (top_h - d)[:, None, None], min=0)).reshape(len(Wt), -1)
            pm = pred - pred.mean(1, keepdim=True)
            s = torch.clamp((pm * xm).sum(1) / ((pm ** 2).sum(1) + 1e-9), min=0)
            e = ((xm - s[:, None] * pm) ** 2).sum(1) / vx_tot              # unexplained fraction
            best_e = torch.minimum(best_e, e)
        # tilted press plane: on the observed contact pixels  X = s * W - (p0 + p1 u + p2 v)  is linear in
        # (s, p0, p1, p2); solve it for every window at once, then score the full model (with the gel blur)
        cm = (xv > 0.05).float()
        if cm.sum() >= 8:
            uu = ((torch.arange(wf, device=self.dev) - (wf - 1) / 2) * RES).repeat(hf)
            vv = ((torch.arange(hf, device=self.dev) - (hf - 1) / 2) * RES).repeat_interleave(wf)
            Wf_ = Wt.reshape(len(Wt), -1)
            cols = [Wf_, -torch.ones_like(Wf_), -uu.expand_as(Wf_), -vv.expand_as(Wf_)]
            A = torch.stack(cols, -1) * cm[None, :, None]                     # (M, N, 4)
            AtA = A.transpose(1, 2) @ A + 1e-6 * torch.eye(4, device=self.dev)
            AtX = (A * (xv * cm)[None, :, None]).sum(1)
            sol = torch.linalg.solve(AtA, AtX[..., None])[..., 0]           # (M, 4)
            sc = torch.clamp(sol[:, 0], min=0.2)
            p1 = torch.clamp(sol[:, 2] / sc, -0.06, 0.06)                    # tilt <= ~3.4 deg
            p2 = torch.clamp(sol[:, 3] / sc, -0.06, 0.06)
            p0 = sol[:, 1] / sc
            plane = p0[:, None] + p1[:, None] * uu[None] + p2[:, None] * vv[None]
            pred = self._blur(torch.clamp(Wf_ - plane, min=0).reshape(len(Wt), hf, wf)).reshape(len(Wt), -1)
            pm = pred - pred.mean(1, keepdim=True)
            s_ = torch.clamp((pm * xm).sum(1) / ((pm ** 2).sum(1) + 1e-9), min=0)
            e = ((xm - s_[:, None] * pm) ** 2).sum(1) / vx_tot
            best_e = torch.minimum(best_e, e)
        e_np = best_e.cpu().numpy()
        cand_best = {}
        for j, (ci, kk, r0, c0) in enumerate(meta):                          # best variant per candidate
            if ci not in cand_best or e_np[j] < cand_best[ci][0]:
                cand_best[ci] = (e_np[j], kk, r0, c0)
        ev = np.array([v[0] for v in cand_best.values()])
        com_np = self.com.cpu().numpy()
        vs = [(((n - 1) / 2 + com_np[kk, 0] / RES - (c0 + (wf - 1) / 2)) * RES,
               ((n - 1) / 2 + com_np[kk, 1] / RES - (r0 + (hf - 1) / 2)) * RES) for (_, kk, r0, c0) in cand_best.values()]
        return dict(contact=True, ncc=best, cmaps=cmaps, e=ev, v=vs,
                    info=dict(peak=peak, area=area, ncc=best, n_cand=len(cand), e_min=float(ev.min())))
