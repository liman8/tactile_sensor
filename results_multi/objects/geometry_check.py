"""Geometry rule check (user rule, 2026-11-22: the STL files will be 3D-printed, so the trained geometry must be the
printed geometry; allowed: unit scaling, rigid transforms, separate CoACD collision parts; any change of the visual mesh
must be reported, and a repair / decimation is allowed only below 0.02 mm deviation, measured and logged).

    python results_multi/objects/geometry_check.py   -> results_multi/objects/geometry_check.json; SHA-256 into objects.json

Per object (the original STL as read from disk, scaled by its unit to mm, in float64):
  (a) triangle-by-triangle: the visual mesh the scene uses (meshes/<name>/<name>_visual.stl) must have the same triangles:
      the triangles are matched as sets of 3 vertices (canonical order), the largest vertex distance is reported, and the
      winding of every matched triangle (same normal direction or flipped);
  (b) surface deviation, both ways: 20,000 points sampled on each surface, distance to the other surface (trimesh
      closest point), max = the symmetric Hausdorff distance (sampled);
  (c) what MuJoCo compiled: the visual geom's vertices in the body frame (gelsight_multi.xml; geom pos / quat applied),
      nearest original vertex, max distance;
  (d) the STL as a file: watertight, triangle count, the scale a slicer would see (STL has no units).
"""
import hashlib
import json
import os
import sys

import numpy as np
import trimesh

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import objects  # noqa: E402
import stl_to_mjcf as s2m  # noqa: E402

TOL_MM = 0.02


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def canon(T):
    """(n, 3, 3) triangles -> rows of 9 numbers with the vertices in lexicographic order."""
    out = np.empty((len(T), 9))
    for i, t in enumerate(T):
        o = np.lexsort(t.T[::-1])
        out[i] = t[o].ravel()
    return out


def main():
    import mujoco
    m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "gelsight_multi.xml"))
    reg = json.load(open("objects.json"))
    rep = {}
    for name in objects.names():
        o = objects.get(name)
        src = os.path.join(ROOT, o["stl"])
        s_mm = s2m.UNITS[o["units"]] * 1000.0
        raw = trimesh.load(src, process=False, force="mesh")
        T0 = np.asarray(raw.triangles, np.float64) * s_mm
        vis = trimesh.load(os.path.join(ROOT, o["mesh_dir"], f"{name}_visual.stl"), process=False, force="mesh")
        T1 = np.asarray(vis.triangles, np.float64)
        r = dict(stl=o["stl"], sha256=sha(src), units=o["units"], scale_to_mm=s_mm, triangles_original=int(len(T0)),
                 triangles_visual=int(len(T1)))
        p0 = trimesh.Trimesh(raw.vertices * s_mm, raw.faces, process=True)
        r["original_watertight"] = bool(p0.is_watertight)
        r["original_extent_in_file_units"] = (raw.extents).round(6).tolist()
        r["printed_size_mm_if_slicer_assumes_mm"] = (raw.extents).round(4).tolist()
        # (a) triangle by triangle
        if len(T0) == len(T1):
            A, B = canon(T0), canon(T1)
            ia, ib = np.lexsort(A.T[::-1]), np.lexsort(B.T[::-1])
            d = np.abs(A[ia] - B[ib]).reshape(len(A), 3, 3)
            r["triangle_match_max_vertex_dev_mm"] = float(np.linalg.norm(d, axis=2).max())
            n0 = np.cross(T0[:, 1] - T0[:, 0], T0[:, 2] - T0[:, 0])
            n1 = np.cross(T1[:, 1] - T1[:, 0], T1[:, 2] - T1[:, 0])
            same = np.sign(np.einsum("ij,ij->i", n0[ia], n1[ib]))
            r["triangles_winding_flipped"] = int((same < 0).sum())
        else:
            r["triangle_match_max_vertex_dev_mm"] = None
        # (b) sampled surface deviation, both ways
        p1 = trimesh.Trimesh(vis.vertices, vis.faces, process=True)
        a = p0.sample(20000, seed=1)
        b = p1.sample(20000, seed=2)
        dab = trimesh.proximity.closest_point(p1, a)[1]
        dba = trimesh.proximity.closest_point(p0, b)[1]
        r["surface_dev_mm"] = dict(visual_to_original_max=float(dba.max()), original_to_visual_max=float(dab.max()),
                                   hausdorff_sampled=float(max(dab.max(), dba.max())))
        # (c) MuJoCo's compiled visual mesh, in the body frame
        mid = m.mesh(f"{name}_visual").id
        gid = [g for g in range(m.ngeom) if m.geom_dataid[g] == mid and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH][0]
        V = np.asarray(m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]], np.float64)
        q = m.geom_quat[gid]
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, q)
        Vb = (V @ R.reshape(3, 3).T + m.geom_pos[gid]) * 1000.0
        from scipy.spatial import cKDTree
        r["mujoco_compiled_vertex_dev_mm"] = float(cKDTree(p0.vertices).query(Vb)[0].max())
        r["mujoco_vertices"] = int(len(V))
        r["visual_mesh_changes"] = json.load(open(os.path.join(ROOT, o["mesh_dir"], "report.json")))["changes"]
        r["passes"] = bool(max(r["surface_dev_mm"]["hausdorff_sampled"], r["mujoco_compiled_vertex_dev_mm"],
                               r["triangle_match_max_vertex_dev_mm"] or 0) < TOL_MM)
        rep[name] = r
        reg["objects"][name]["stl_sha256"] = r["sha256"]
        reg["objects"][name]["geometry_check"] = dict(
            rule="visual mesh = the original STL (unit scaling + rigid transforms only); deviation < 0.02 mm",
            triangles=r["triangles_original"], same_triangles=r["triangle_match_max_vertex_dev_mm"] is not None,
            max_vertex_dev_mm=r["triangle_match_max_vertex_dev_mm"], winding_flipped=r.get("triangles_winding_flipped"),
            surface_hausdorff_mm=r["surface_dev_mm"]["hausdorff_sampled"],
            mujoco_compiled_dev_mm=r["mujoco_compiled_vertex_dev_mm"], visual_mesh_changes=r["visual_mesh_changes"],
            passes=r["passes"], file="results_multi/objects/geometry_check.json")
        print(f"{name:10s} {o['stl']:18s} sha256 {r['sha256'][:12]}  triangles {r['triangles_original']} -> {r['triangles_visual']}, "
              f"vertex dev {r['triangle_match_max_vertex_dev_mm']:.2e} mm, flipped {r.get('triangles_winding_flipped')}, "
              f"surface {r['surface_dev_mm']['hausdorff_sampled']:.2e} mm, MuJoCo {r['mujoco_compiled_vertex_dev_mm']:.2e} mm, "
              f"watertight {r['original_watertight']} -> {'PASS' if r['passes'] else 'FAIL'}")
    json.dump(rep, open(os.path.join(ROOT, "results_multi", "objects", "geometry_check.json"), "w"), indent=1)
    json.dump(reg, open("objects.json", "w"), indent=1)


if __name__ == "__main__":
    main()
