"""Second look: true dimensions (oriented box), the spoon's extra components, quasi-static stable poses (trimesh).
    .venv/bin/python -I results_multi/ycb/inspect_ycb2.py downloads/ycb"""
import os
import sys

import numpy as np
import trimesh

root = sys.argv[1]
for o in sorted(os.listdir(root)):
    m = trimesh.load(os.path.join(root, o, o, "google_16k", "nontextured.stl"), process=True)
    m.apply_scale(1000.0)                                                  # metres -> mm
    obb = np.sort(m.bounding_box_oriented.primitive.extents)[::-1]
    comps = m.split(only_watertight=False)
    big = max(comps, key=lambda c: len(c.faces))
    small = [(len(c.faces), float(np.linalg.norm(c.extents))) for c in comps if c is not big]
    try:
        T, p = trimesh.poses.compute_stable_poses(big if big.is_watertight else m.convex_hull, n_samples=2000, threshold=0.0)
        top = sorted(p, reverse=True)
        nat = [float(x) for x in top if x >= 0.01]
    except Exception as ex:
        nat = f"failed: {ex}"
    print(f"{o:26s} size {obb.round(1)} mm | components {len(comps)} (others: faces, diag mm {small}) | "
          f"stable poses >= 1 %: {len(nat) if isinstance(nat, list) else nat} {np.round(nat, 3).tolist() if isinstance(nat, list) else ''}", flush=True)
