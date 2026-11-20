"""
Fork recognition from GelSight-style tactile images.

It only uses what a real sensor gives you: the current RGB frame and a no-contact
reference frame. The pipeline follows the usual GelSight processing:

  1. difference image  I - I_ref, markers removed (detected as dark dots, filled
     by normalised convolution)
  2. photometric stereo: dI = A (n - n0) + b  ->  surface normal n per pixel.
     A, b are fitted by `calibrate()` from presses of a 5 mm ball bearing,
     the standard GelSight calibration
  3. Poisson integration (DCT, Neumann boundary)  ->  height map in mm
  4. contact mask + rotation-invariant shape features: how many parallel "runs"
     each scan line crosses after aligning with the contact's principal axis
     (a fork gives 2-4 regularly spaced prongs), run widths and gaps, solidity,
     convexity defects, Hu moments, ...
  5. RandomForest  ->  P(fork)

    det = ForkDetector.load("fork_detector.joblib")
    res = det.detect(rgb, ref)
    res.is_fork, res.p_fork, res.n_tines, res.tine_angle_deg, res.centroid_mm
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter

FEATURE_NAMES: list[str] = []   # filled on first call of shape_features()


# ---------------------------------------------------------------------- reconstruction
def remove_markers(img: np.ndarray, ref: np.ndarray, sigma_px: float = 2.0):
    """Return (dI float HxWx3, marker_mask) with marker pixels filled in."""
    I = img.astype(np.float32)
    R = ref.astype(np.float32)

    def dark(x):
        lum = x.sum(-1)
        bg = cv2.medianBlur(np.clip(x, 0, 255).astype(np.uint8), 11).astype(np.float32).sum(-1)
        return lum < 0.6 * bg

    mask = (dark(I) | dark(R)).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    dI = I - R
    w = (~mask).astype(np.float32)
    num = np.stack([gaussian_filter(dI[..., c] * w, sigma_px) for c in range(3)], -1)
    den = gaussian_filter(w, sigma_px)[..., None] + 1e-6
    filled = np.where(mask[..., None], num / den, dI)
    return filled, mask


def poisson_neumann(gx: np.ndarray, gy: np.ndarray, baseline_pct: float = 20.0) -> np.ndarray:
    """Integrate a gradient field (per-pixel units), Neumann boundary (DCT solver).

    Unlike the zero-boundary (DST) solver used in many GelSight codes, this does not
    pull contacts that run off the edge of the pad down to zero. The free constant
    is fixed by assuming at least `baseline_pct` % of the pad is untouched."""
    H, W = gx.shape
    gx = gx.copy()
    gy = gy.copy()
    gx[:, -1] = 0
    gy[-1, :] = 0
    f = np.zeros_like(gx)
    f[:, 0] += gx[:, 0]
    f[:, 1:] += gx[:, 1:] - gx[:, :-1]
    f[0, :] += gy[0, :]
    f[1:, :] += gy[1:, :] - gy[:-1, :]
    i = np.arange(H)[:, None]
    j = np.arange(W)[None, :]
    denom = (2 * np.cos(np.pi * i / H) - 2) + (2 * np.cos(np.pi * j / W) - 2)
    denom[0, 0] = 1.0
    F = dctn(f, type=2, norm="ortho") / denom
    F[0, 0] = 0.0
    h = idctn(F, type=2, norm="ortho")
    return h - np.percentile(h, baseline_pct)


class PhotometricModel:
    """dI_c = A_c . (n - n0) + b_c   (one linear model per colour channel)."""

    def __init__(self, A=None, b=None):
        self.A = None if A is None else np.asarray(A, float)
        self.b = np.zeros(3) if b is None else np.asarray(b, float)

    # -- calibration with a ball of known radius ---------------------------------
    def calibrate_ball(self, images, ref, ball_radius_mm: float, mm_per_px: float):
        X_all, Y_all = [], []
        for img in images:
            dI, mask = remove_markers(img, ref)
            dI = gaussian_filter(dI, (0.8, 0.8, 0))
            mag = np.linalg.norm(dI, axis=-1)
            bw = (mag > max(8.0, 0.25 * mag.max())).astype(np.uint8)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            (cx, cy), a_px = cv2.minEnclosingCircle(c)
            a_mm = min(a_px * mm_per_px, 0.98 * ball_radius_mm)
            H, W = mag.shape
            yy, xx = np.mgrid[0:H, 0:W]
            x_mm, y_mm = (xx - cx) * mm_per_px, (yy - cy) * mm_per_px
            r = np.hypot(x_mm, y_mm)
            inside = (r < 0.85 * a_mm) & ~mask
            outside = (r > 1.6 * a_mm + 1.0) & ~mask
            R = ball_radius_mm
            nz = np.sqrt(np.clip(R**2 - r**2, 1e-9, None))
            n = np.stack([x_mm / R, y_mm / R, nz / R], -1)
            n[outside] = (0, 0, 1)
            sel = inside | (outside & (np.random.default_rng(0).random(outside.shape) < 0.02))
            nn = n[sel]
            X_all.append(np.column_stack([nn[:, 0], nn[:, 1], nn[:, 2] - 1]))
            Y_all.append(dI[sel])
        X = np.concatenate(X_all)
        Y = np.concatenate(Y_all)
        # no intercept: an unchanged image (dI = 0) must map to an undeformed, flat gel
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)          # (3, 3)
        self.A = coef.T                                       # rows: channel
        self.b = np.zeros(3)
        pred = X @ coef
        return float(np.sqrt(np.mean((pred - Y) ** 2)))       # RMS fit error (grey levels)

    # -- reconstruction ------------------------------------------------------------
    def height_map(self, img, ref, mm_per_px: float, max_slope: float = 3.0,
                   flat_level: float = 5.0):
        """Returns (height mm, gx, gy, |dI|). Pixels whose colour barely changed
        (|dI| < flat_level grey levels) are used to remove any global tilt
        (calibration bias, slow illumination drift) before integration."""
        dI, _ = remove_markers(img, ref)
        dI = gaussian_filter(dI, (0.8, 0.8, 0))
        mag = np.linalg.norm(dI, axis=-1)
        dn = (dI - self.b) @ np.linalg.inv(self.A).T
        n = dn + np.array([0.0, 0.0, 1.0])
        nz = np.clip(n[..., 2], 0.25, None)
        gx = np.clip(-n[..., 0] / nz, -max_slope, max_slope)
        gy = np.clip(-n[..., 1] / nz, -max_slope, max_slope)
        flat = mag < flat_level
        if flat.sum() > 0.05 * flat.size:
            gx = gx - np.median(gx[flat])
            gy = gy - np.median(gy[flat])
        h = poisson_neumann(gx * mm_per_px, gy * mm_per_px)
        return h, gx, gy, mag


# ---------------------------------------------------------------------- shape features
def contact_mask(h: np.ndarray, abs_thr_mm: float = 0.06, rel_thr: float = 0.3,
                 min_area_px: int = 30):
    peak = np.percentile(h, 99.9)
    thr = max(abs_thr_mm, rel_thr * peak)
    m = (h > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_px
    return keep[lab], thr


def _runs(line: np.ndarray, min_run: int, min_gap: int):
    """Runs of True in a 1-D bool array after closing small gaps / dropping slivers."""
    if not line.any():
        return [], []
    d = np.diff(np.concatenate([[0], line.astype(np.int8), [0]]))
    starts, ends = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
    # merge runs separated by tiny gaps
    s, e = [starts[0]], [ends[0]]
    for a, b in zip(starts[1:], ends[1:]):
        if a - e[-1] < min_gap:
            e[-1] = b
        else:
            s.append(a)
            e.append(b)
    s, e = np.array(s), np.array(e)
    keep = (e - s) >= min_run
    s, e = s[keep], e[keep]
    widths = e - s
    gaps = s[1:] - e[:-1]
    return widths.tolist(), gaps.tolist()


def _scan_stats(mask: np.ndarray, axis: int, mm: float, min_run: int, min_gap: int):
    """Scan lines along `axis` (0: columns, 1: rows); summarise run structure."""
    lines = mask.T if axis == 0 else mask
    ks, W_multi, G_multi, nz = [], [], [], 0
    multi_idx = []
    for idx, ln in enumerate(lines):
        if not ln.any():
            continue
        nz += 1
        w, g = _runs(ln, min_run, min_gap)
        ks.append(len(w))
        if len(w) >= 2:
            W_multi += w
            G_multi += g
            multi_idx.append(idx)
    ks = np.array(ks) if ks else np.zeros(1)
    multi = ks >= 2
    f = {}
    f["n_lines"] = float(nz)
    f["frac_k2"] = float(multi.mean()) if nz else 0.0
    f["frac_k3"] = float((ks >= 3).mean()) if nz else 0.0
    f["frac_k4"] = float((ks >= 4).mean()) if nz else 0.0
    f["frac_k6"] = float((ks >= 6).mean()) if nz else 0.0
    f["len_k2_mm"] = float(multi.sum() * mm)
    if multi.any():
        km = ks[multi]
        f["k_mode"] = float(np.bincount(km.astype(int)).argmax())
        f["k_max"] = float(km.max())
        W, G = np.array(W_multi) * mm, np.array(G_multi) * mm
        f["w_med"] = float(np.median(W))
        f["w_cv"] = float(W.std() / (W.mean() + 1e-9))
        f["g_med"] = float(np.median(G)) if len(G) else 0.0
        f["g_cv"] = float(G.std() / (G.mean() + 1e-9)) if len(G) else 0.0
        f["pitch_med"] = f["w_med"] + f["g_med"]
        # longest contiguous stretch of multi-run lines (= visible prong length)
        idx = np.array(multi_idx)
        brk = np.nonzero(np.diff(idx) > 2)[0]
        seg = np.split(idx, brk + 1)
        f["run_len_mm"] = float(max(len(s_) for s_ in seg) * mm)
    else:
        for k in ["k_mode", "k_max", "w_med", "w_cv", "g_med", "g_cv", "pitch_med", "run_len_mm"]:
            f[k] = 0.0
    return f


def shape_features(mask: np.ndarray, h: np.ndarray, mm_per_px: float):
    """Rotation-invariant features + geometry info (tine direction etc.)."""
    global FEATURE_NAMES
    mm = mm_per_px
    feats, info = {}, {}
    area_px = int(mask.sum())
    feats["area_mm2"] = area_px * mm * mm
    hv = h[mask] if area_px else np.zeros(1)
    feats["h_max"] = float(hv.max())
    feats["h_mean"] = float(hv.mean())
    feats["h_std_rel"] = float(hv.std() / (hv.mean() + 1e-9))
    m8 = mask.astype(np.uint8)
    n_comp, lab, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    feats["n_comp"] = float(n_comp - 1)

    ys, xs = np.nonzero(mask)
    if area_px < 5:
        theta, elong = 0.0, 1.0
        cx = cy = 0.0
    else:
        cx, cy = xs.mean(), ys.mean()
        cov = np.cov(np.stack([xs - cx, ys - cy]))
        evals, evecs = np.linalg.eigh(cov)
        v = evecs[:, 1]
        theta = float(np.arctan2(v[1], v[0]))
        elong = float(np.sqrt(evals[1] / max(evals[0], 1e-6)))
    feats["elongation"] = min(elong, 50.0)
    info["centroid_px"] = (float(cx), float(cy))

    # solidity + convexity defects over all contact pixels together
    if area_px >= 5:
        pts = np.column_stack([xs, ys]).astype(np.int32)
        hull = cv2.convexHull(pts)
        hull_area = max(cv2.contourArea(hull), 1.0)
        feats["solidity"] = float(min(area_px / hull_area, 1.0))
        cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        big = max(cnts, key=cv2.contourArea)
        ndef = 0
        if len(big) >= 5:
            hi = cv2.convexHull(big, returnPoints=False)
            try:
                defects = cv2.convexityDefects(big, hi)
            except cv2.error:
                defects = None
            if defects is not None and len(defects):
                defects = np.asarray(defects).reshape(-1, 4)       # OpenCV 4: (N,1,4), 5: (N,4)
                depth_mm = defects[:, 3] / 256.0 * mm
                ndef = int((depth_mm > 1.5).sum())
        feats["n_defects"] = float(ndef)
        hu = cv2.HuMoments(cv2.moments(m8, binaryImage=True)).ravel()
        for i, v_ in enumerate(hu):
            feats[f"hu{i}"] = float(-np.sign(v_) * np.log10(abs(v_) + 1e-30))
    else:
        feats["solidity"] = 1.0
        feats["n_defects"] = 0.0
        for i in range(7):
            feats[f"hu{i}"] = 0.0

    # rotate so the principal axis is horizontal, then scan both ways
    H, W = mask.shape
    D = int(np.ceil(np.hypot(H, W))) + 4
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), np.degrees(theta), 1.0)
    M[:, 2] += (D / 2 - cx, D / 2 - cy)
    rot = cv2.warpAffine(m8, M, (D, D), flags=cv2.INTER_NEAREST).astype(bool)
    min_run, min_gap = max(2, int(round(0.3 / mm))), max(2, int(round(0.4 / mm)))
    s_perp = _scan_stats(rot, axis=0, mm=mm, min_run=min_run, min_gap=min_gap)   # columns
    s_par = _scan_stats(rot, axis=1, mm=mm, min_run=min_run, min_gap=min_gap)    # rows
    # "comb direction" = the scan direction that crosses more multi-run lines
    if s_perp["len_k2_mm"] >= s_par["len_k2_mm"]:
        best, other, tine_axis = s_perp, s_par, theta            # prongs along principal axis
    else:
        best, other, tine_axis = s_par, s_perp, theta + np.pi / 2
    for k, v_ in best.items():
        feats[f"c_{k}"] = v_
    for k in ["frac_k2", "frac_k3", "k_mode", "len_k2_mm", "w_med", "g_med"]:
        feats[f"o_{k}"] = other[k]
    info["tine_axis_rad"] = float(tine_axis)
    info["k_mode"] = int(best["k_mode"])
    if not FEATURE_NAMES:
        FEATURE_NAMES = list(feats.keys())
    return np.array([feats[k] for k in FEATURE_NAMES], float), info


def edge_axis(mask: np.ndarray) -> float:
    """Dominant edge orientation of a contact mask (structure tensor), returned as the
    direction ALONG the edges. For a fork the long tine edges dominate, so this is the
    tine axis; it stays right when the palm or only a corner of the fork is on the pad,
    where the blob's principal axis does not (median error ~3 deg vs ~23 deg)."""
    sm = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0)
    ey, ex = np.gradient(sm)
    phi = 0.5 * np.arctan2((2 * ex * ey).sum(), (ex * ex - ey * ey).sum())
    return float(phi + np.pi / 2)


def count_tines(h: np.ndarray, tine_axis_rad: float, mm_per_px: float,
                rel_thr: float = 0.15, min_len_mm: float = 0.5) -> int:
    """Prongs seen side by side: the largest k such that scan lines across the tines
    cross >= k separate contacts over at least `min_len_mm` of tine length. Uses a
    lower depth threshold than the classifier mask so lightly touching tines count."""
    mask, _ = contact_mask(h, abs_thr_mm=0.05, rel_thr=rel_thr)
    if mask.sum() < 5:
        return 0
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    H, W = mask.shape
    D = int(np.ceil(np.hypot(H, W))) + 4
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), np.degrees(tine_axis_rad), 1.0)
    M[:, 2] += (D / 2 - cx, D / 2 - cy)
    rot = cv2.warpAffine(mask.astype(np.uint8), M, (D, D), flags=cv2.INTER_NEAREST).astype(bool)
    min_run, min_gap = max(2, int(round(0.3 / mm_per_px))), max(2, int(round(0.4 / mm_per_px)))
    ks = np.array([len(_runs(c, min_run, min_gap)[0]) for c in rot.T if c.any()])
    L = max(1, int(round(min_len_mm / mm_per_px)))
    best = 0
    for k in range(1, 12):
        if (ks >= k).sum() >= L:
            best = k
    return best


# ---------------------------------------------------------------------- detector
@dataclass
class Detection:
    contact: bool
    is_fork: bool
    p_fork: float
    n_tines: int                 # tines seen side by side on the pad (approx., +-1), 0 if not a fork
    tine_angle_deg: float        # tine axis in the image/gel frame, mod 180
    centroid_mm: tuple           # contact centroid (x, y) relative to gel centre
    peak_depth_mm: float
    contact_area_mm2: float

    def as_dict(self):
        return asdict(self)


class ForkDetector:
    def __init__(self, mm_per_px: float = 0.1, photometric: PhotometricModel | None = None,
                 clf=None, threshold: float = 0.5, contact_depth_mm: float = 0.06,
                 contact_area_mm2: float = 0.5, rel_thr: float = 0.3, count_feature: bool = True,
                 evidence_level: float = 8.0, evidence_mm2: float = 0.3):
        self.evidence_level = evidence_level    # grey levels of |I - I_ref|
        self.evidence_mm2 = evidence_mm2
        self.mm_per_px = mm_per_px
        self.rel_thr = rel_thr
        self.count_feature = count_feature
        self.pm = photometric or PhotometricModel()
        self.clf = clf
        self.threshold = threshold
        self.contact_depth_mm = contact_depth_mm
        self.contact_area_mm2 = contact_area_mm2

    # -------------------------------------------------------------- pieces
    def calibrate(self, ball_images, ref, ball_radius_mm: float = 2.5):
        return self.pm.calibrate_ball(ball_images, ref, ball_radius_mm, self.mm_per_px)

    def reconstruct(self, img, ref):
        return self.pm.height_map(img, ref, self.mm_per_px)[0]

    def features(self, img, ref):
        h, _, _, mag = self.pm.height_map(img, ref, self.mm_per_px)
        mask, _ = contact_mask(h, abs_thr_mm=self.contact_depth_mm, rel_thr=self.rel_thr)
        f, info = shape_features(mask, h, self.mm_per_px)
        # direct photometric evidence of contact: pixels whose colour clearly changed
        info["evidence_mm2"] = float((mag > self.evidence_level).sum()) * self.mm_per_px ** 2
        info["area_mm2"] = float(mask.sum()) * self.mm_per_px ** 2
        info["n_tines"] = count_tines(h, info["tine_axis_rad"], self.mm_per_px)
        low, _ = contact_mask(h, abs_thr_mm=0.05, rel_thr=0.15)
        info["tine_angle_rad"] = edge_axis(low) if low.any() else float("nan")
        if self.count_feature:
            f = np.append(f, info["n_tines"])      # extra feature: low-threshold prong count
        return f, info, h, mask

    # -------------------------------------------------------------- training
    def fit(self, X, y, **rf_kwargs):
        from sklearn.ensemble import RandomForestClassifier
        params = dict(n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                      n_jobs=-1, random_state=0)
        params.update(rf_kwargs)
        self.clf = RandomForestClassifier(**params).fit(X, y)
        return self

    # -------------------------------------------------------------- inference
    def in_contact(self, info) -> bool:
        return info["area_mm2"] >= self.contact_area_mm2 and info["evidence_mm2"] >= self.evidence_mm2

    def detect(self, img, ref) -> Detection:
        f, info, h, mask = self.features(img, ref)
        area = info["area_mm2"]
        peak = float(h[mask].max()) if mask.any() else 0.0
        cx, cy = info["centroid_px"]
        H, W = mask.shape
        cmm = ((cx - (W - 1) / 2) * self.mm_per_px, (cy - (H - 1) / 2) * self.mm_per_px)
        if not self.in_contact(info):
            return Detection(False, False, 0.0, 0, float("nan"), cmm, peak, area)
        p = float(self.clf.predict_proba(f[None])[0, 1])
        is_fork = p >= self.threshold
        ang = float(np.degrees(info["tine_angle_rad"]) % 180.0)
        return Detection(True, is_fork, p, info["n_tines"] if is_fork else 0,
                         ang if is_fork else float("nan"), cmm, peak, area)

    # -------------------------------------------------------------- persistence
    def save(self, path):
        import joblib
        joblib.dump(dict(mm_per_px=self.mm_per_px, A=self.pm.A, b=self.pm.b, clf=self.clf,
                         threshold=self.threshold, contact_depth_mm=self.contact_depth_mm,
                         contact_area_mm2=self.contact_area_mm2, rel_thr=self.rel_thr,
                         count_feature=self.count_feature, evidence_level=self.evidence_level,
                         evidence_mm2=self.evidence_mm2,
                         feature_names=FEATURE_NAMES), path)

    @classmethod
    def load(cls, path):
        import joblib
        global FEATURE_NAMES
        d = joblib.load(path)
        FEATURE_NAMES = list(d["feature_names"])
        return cls(d["mm_per_px"], PhotometricModel(d["A"], d["b"]), d["clf"], d["threshold"],
                   d["contact_depth_mm"], d["contact_area_mm2"], d.get("rel_thr", 0.3),
                   d.get("count_feature", False), d.get("evidence_level", 8.0), d.get("evidence_mm2", 0.3))


def overlay(img_rgb: np.ndarray, det: Detection, mm_per_px: float = 0.1, scale: int = 2):
    """Annotated BGR image for display."""
    out = cv2.resize(img_rgb[..., ::-1], None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    H, W = img_rgb.shape[:2]
    if not det.contact:
        label, col = "no contact", (200, 200, 200)
    elif det.is_fork:
        label, col = f"FORK  p={det.p_fork:.2f}  tines={det.n_tines}", (60, 255, 60)
        cx = (det.centroid_mm[0] / mm_per_px + (W - 1) / 2) * scale
        cy = (det.centroid_mm[1] / mm_per_px + (H - 1) / 2) * scale
        a = np.radians(det.tine_angle_deg)
        L = 60 * scale / 2
        p1 = (int(cx - L * np.cos(a)), int(cy - L * np.sin(a)))
        p2 = (int(cx + L * np.cos(a)), int(cy + L * np.sin(a)))
        cv2.line(out, p1, p2, col, 2, cv2.LINE_AA)
    else:
        label, col = f"not a fork  p={det.p_fork:.2f}", (60, 60, 255)
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
    return out
