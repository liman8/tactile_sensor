"""Printability check (addendum B): minimum wall / feature thickness of each STL as it will be printed (in mm; the
geometry is not modified).

    python results_multi/objects/printability.py   -> results_multi/objects/printability.json, printability_<name>.png

Two measures:
  (1) 3-D wall thickness: 60,000 points on the surface (area-weighted) plus every vertex; from each, a ray along the
      inward normal to the next surface of the mesh. On a plate this is the plate thickness from the top / bottom faces
      and the in-plane width from the side walls.
  (2) In-plane feature width (all four objects are long and thin in the STL x-y plane): the 2-D LOCAL THICKNESS of the
      top-view outline (raster 0.02 mm): at each point, the diameter of the largest disc inside the outline that
      contains the point (Hildebrand & Rueegsegger), computed in 0.05 mm levels by dilating {distance >= r} with a disc of
      radius r. A rounded end of a 4 mm handle keeps 4 mm; a pointed tip goes to 0 at its very end.
      (A first version took column-wise maxima of the distance transform; it reported the rounded handle end as thin,
      so it was replaced, RUN_LOG.)
  A feature is thinner than t if (2) < t there, or if (1) < t at points away from a sharp tip. Each region under
  1.0 mm is listed with its STL x range, part and minimum; under 0.8 mm is flagged (put first in the final report).
"""
import json
import os
import sys

import cv2
import numpy as np
import trimesh

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import objects  # noqa: E402
import stl_to_mjcf as s2m  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
PPM = 50.0         # 0.02 mm raster


def ray_thickness(m, n=60000):
    pts, fi = trimesh.sample.sample_surface(m, n, seed=0)
    P = np.vstack([pts, m.vertices])
    N = np.vstack([m.face_normals[fi], m.vertex_normals])
    O = P - N * 1e-4
    loc, idx, _ = m.ray.intersects_location(O, -N, multiple_hits=False)
    t = np.full(len(P), np.nan)
    t[idx] = np.linalg.norm(loc - O[idx], axis=1)
    return P, t


def footprint(m):
    T = m.triangles[:, :, :2]
    lo = m.bounds[0, :2] - 1.0
    hi = m.bounds[1, :2] + 1.0
    W, H = int(np.ceil((hi[0] - lo[0]) * PPM)) + 1, int(np.ceil((hi[1] - lo[1]) * PPM)) + 1
    mask = np.zeros((H, W), np.uint8)
    px = (T - lo) * PPM
    for t in px:
        cv2.fillPoly(mask, [np.round(t * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return mask, lo


def inplane_width(m, levels=np.arange(0.05, 2.51, 0.05)):
    """2-D local thickness map (mm) of the top-view outline, and the per-column minimum along STL x."""
    mask, lo = footprint(m)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) / PPM
    lt = np.zeros(mask.shape, np.float32)
    for r in levels:
        core = (dt >= r).astype(np.uint8)
        if not core.any():
            break
        k = int(np.ceil(r * PPM))
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        cover = cv2.dilate(core, disc) > 0
        lt[cover & (mask > 0)] = 2 * r
    lt[(mask > 0) & (lt == 0)] = 0.05                         # thinner than the smallest level
    xs = lo[0] + np.arange(mask.shape[1]) / PPM
    colmin = np.where((mask > 0).any(0), np.where(mask > 0, lt, np.inf).min(0), np.nan)
    return xs, colmin, mask, lt, lo


def regions(xs, w, thr, name):
    """Contiguous x ranges where the in-plane width is below thr."""
    bad = np.isfinite(w) & (w < thr)
    out = []
    i = 0
    while i < len(xs):
        if bad[i]:
            j = i
            while j + 1 < len(xs) and bad[j + 1]:
                j += 1
            x0, x1 = float(xs[i]), float(xs[j])
            part = objects.part_names(name)[int(objects.part_from_x(objects.part_starts_mm(name), (x0 + x1) / 2))]
            out.append(dict(x_from_mm=round(x0, 2), x_to_mm=round(x1, 2), length_mm=round(x1 - x0 + 1 / PPM, 2),
                            part=part, min_width_mm=round(float(np.nanmin(w[i:j + 1])), 3)))
            i = j + 1
        else:
            i += 1
    return out


def main():
    rep = {}
    for name in objects.names():
        o = objects.get(name)
        raw = trimesh.load(os.path.join(ROOT, o["stl"]), process=False, force="mesh")
        m = trimesh.Trimesh(raw.vertices * s2m.UNITS[o["units"]] * 1000.0, raw.faces, process=True)
        P, t = ray_thickness(m)
        xs, w, mask, lt, lo = inplane_width(m)
        # wall thickness from the top / bottom faces only (|normal z| > 0.9) and at least 1 mm inside the outline (a ray
        # from a point next to the edge leaves through the side wall): the sheet / plate thickness
        Pn = m.nearest.on_surface(P)[2]
        nz = np.abs(m.face_normals[Pn][:, 2])
        dt2 = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) / PPM
        ij = np.clip(np.round((P[:, :2] - lo) * PPM).astype(int), 0, [mask.shape[1] - 1, mask.shape[0] - 1])
        inner = dt2[ij[:, 1], ij[:, 0]] >= 1.0
        sheet = t[(nz > 0.9) & inner & np.isfinite(t)]
        side = t[(nz < 0.3) & np.isfinite(t)]
        r = dict(stl=o["stl"], sha256=o.get("stl_sha256"), units_scaled_to_mm=o["units"] != "mm",
                 sheet_thickness_mm=dict(min=float(np.min(sheet)), p01=float(np.percentile(sheet, 1)),
                                         median=float(np.median(sheet))),
                 side_ray_width_mm=dict(min=float(np.min(side)), p01=float(np.percentile(side, 1))),
                 inplane_width_min_mm=float(np.nanmin(w)), inplane_width_min_at_x_mm=float(xs[int(np.nanargmin(w))]),
                 below_1_0_mm=regions(xs, w, 1.0, name), below_0_8_mm=regions(xs, w, 0.8, name))
        # a tip that ends in a point is thin at its very end by definition; report how long the thin part is
        # the sheet thickness along the object (rays from the top / bottom faces >= 1 mm inside the outline), by x region;
        # the 5th percentile is the robust minimum (single rays near a side wall or a bend leave early)
        edges = [o["bbox_mm"][0][0]] + sorted(set([-15.0, 1.0, 7.0, 15.0, 25.0, 33.0])) + [o["bbox_mm"][1][0] + 0.01]
        r["sheet_by_x"] = []
        for x0, x1 in zip(edges[:-1], edges[1:]):
            sel = (nz > 0.9) & inner & np.isfinite(t) & (P[:, 0] >= x0) & (P[:, 0] < x1)
            if sel.sum() > 20:
                r["sheet_by_x"].append(dict(x_from_mm=round(x0, 1), x_to_mm=round(x1, 1), median_mm=round(float(np.median(t[sel])), 3),
                                            p05_mm=round(float(np.percentile(t[sel], 5)), 3)))
        # width of each separate piece across the object (tines) near the head end
        r["piece_widths_mm"] = []
        for x in np.arange(np.floor(o["bbox_mm"][1][0]) - 16, o["bbox_mm"][1][0], 2.0).tolist() + [o["bbox_mm"][1][0] - 0.1]:
            c = int(round((x - lo[0]) * PPM))
            v = mask[:, c].astype(int)
            d = np.diff(np.r_[0, v, 0])
            r["piece_widths_mm"].append(dict(x_mm=round(float(x), 2), widths=[round((b - a) / PPM, 2) for a, b in
                                                                               zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0])]))
        r["sharp_corners_note"] = ("the 2-D local thickness goes to 0 at every sharp (non-rounded) corner of the outline; the "
                                   "regions listed under below_*_mm that are 0.3 mm long at x = 36.0-36.3 are the two corners "
                                   "of each square tine end, not a thin feature: the tine itself is >= the piece width there")
        thin_sheet = [x for x in r["sheet_by_x"] if x["p05_mm"] < 1.0]
        min_piece = min(min(pw["widths"]) for pw in r["piece_widths_mm"] if pw["widths"])
        r["features_below_1_0_mm"] = ([f"sheet {x['p05_mm']:.2f}-{x['median_mm']:.2f} mm thick at x {x['x_from_mm']}..{x['x_to_mm']} mm "
                                       f"({objects.part_names(name)[int(objects.part_from_x(objects.part_starts_mm(name), (x['x_from_mm'] + x['x_to_mm']) / 2))]})"
                                       for x in thin_sheet]
                                      + ([f"piece (tine) width {min_piece:.2f} mm near the head end"] if min_piece < 1.0 else []))
        r["features_below_0_8_mm"] = ([f"sheet {x['p05_mm']:.2f} mm at x {x['x_from_mm']}..{x['x_to_mm']}" for x in thin_sheet if x["p05_mm"] < 0.8]
                                      + ([f"piece width {min_piece:.2f} mm"] if min_piece < 0.8 else []))
        r["min_feature_thickness_mm"] = float(min(min(x["p05_mm"] for x in r["sheet_by_x"]), min_piece))
        r["min_feature_note"] = "min over the sheet thickness (5th percentile per x region) and the piece widths; sharp corners excluded"
        rep[name] = r
        # figure: the outline coloured by local width (blue thick ... red < 1 mm)
        v = np.clip(lt / 4.0, 0, 1)
        col = cv2.applyColorMap((255 - v * 255).astype(np.uint8), cv2.COLORMAP_JET)
        col[mask == 0] = 30
        thin = (lt < 1.0) & (mask > 0)
        col[thin] = (0, 0, 255)
        small = cv2.resize(col[::-1], (col.shape[1] // 5, col.shape[0] // 5), interpolation=cv2.INTER_AREA)
        bar = np.zeros((26, small.shape[1], 3), np.uint8)
        cv2.putText(bar, f"{name} ({o['stl']}): local width (blue = 4 mm+, red = under 1.0 mm); sheet {r['sheet_thickness_mm']['median']:.2f} mm",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(OUT, f"printability_{name}.png"), np.concatenate([bar, small], 0))
        print(f"{name:10s} min feature {r['min_feature_thickness_mm']:.2f} mm | < 1.0 mm: {r['features_below_1_0_mm']} | < 0.8 mm: "
              f"{r['features_below_0_8_mm']} | sheet by x {[(x['x_from_mm'], x['x_to_mm'], x['p05_mm']) for x in r['sheet_by_x']]} | "
              f"widths near the head {[(pw['x_mm'], pw['widths']) for pw in r['piece_widths_mm'][-3:]]}", flush=True)
    json.dump(rep, open(os.path.join(OUT, "printability.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
