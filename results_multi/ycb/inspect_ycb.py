"""Inspect downloaded YCB google_16k meshes against the geometry rule (watertight, <= 100k triangles) before import.
    .venv/bin/python -I results_multi/ycb/inspect_ycb.py downloads/ycb   -> results_multi/ycb/inspect_ycb.json
The meshes are read as data only (trimesh); nothing from the download folder is executed."""
import hashlib
import json
import os
import sys

import numpy as np
import trimesh

root = sys.argv[1]
out = {}
for o in sorted(os.listdir(root)):
    f = os.path.join(root, o, o, "google_16k", "nontextured.stl")
    raw = open(f, "rb").read()
    m = trimesh.load(f, process=False)
    mp = trimesh.load(f, process=True)                       # merged vertices (the STL repeats them per triangle)
    ext = m.extents
    comps = mp.split(only_watertight=False)
    e = mp.edges_sorted
    _, cnt = np.unique(e, axis=0, return_counts=True)
    out[o] = dict(sha256=hashlib.sha256(raw).hexdigest(), triangles=int(len(m.faces)),
                  extents_file_units=[float(v) for v in ext], longest_mm_if_m=float(ext.max() * 1000),
                  watertight=bool(mp.is_watertight), winding_consistent=bool(mp.is_winding_consistent),
                  components=len(comps), boundary_edges=int((cnt == 1).sum()), nonmanifold_edges=int((cnt > 2).sum()),
                  degenerate_faces=int((mp.area_faces < 1e-14).sum()),
                  volume_cm3_if_m=float(mp.volume * 1e6) if mp.is_watertight else None,
                  euler=int(mp.euler_number))
    print(o, out[o], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "inspect_ycb.json"), "w"), indent=1)
