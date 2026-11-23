"""Can YCB 031_spoon be made watertight within 0.02 mm (geometry rule)? Drop the tiny extra components, check the main
one; deviation = the largest distance from any removed vertex to the kept surface.
    .venv/bin/python -I results_multi/ycb/spoon_repair_check.py downloads/ycb/031_spoon/031_spoon/google_16k/nontextured.stl"""
import sys

import numpy as np
import trimesh

m = trimesh.load(sys.argv[1], process=True)
m.apply_scale(1000.0)
comps = m.split(only_watertight=False)
big = max(comps, key=lambda c: len(c.faces))
rest = [c for c in comps if c is not big]
_, d, _ = trimesh.proximity.closest_point(big, np.vstack([c.vertices for c in rest]))
e = big.edges_sorted
_, cnt = np.unique(e, axis=0, return_counts=True)
print(f"main component: faces {len(big.faces)}, watertight {big.is_watertight}, boundary edges {(cnt == 1).sum()}, "
      f"non-manifold edges {(cnt > 2).sum()}, degenerate faces {(big.area_faces < 1e-8).sum()}")
print(f"removed: {len(rest)} components, {sum(len(c.faces) for c in rest)} faces; largest distance of a removed vertex "
      f"to the kept surface {d.max():.4f} mm")
