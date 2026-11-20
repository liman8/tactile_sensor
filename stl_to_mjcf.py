"""
Turn an STL into a GelSight object for gelsight_moving.xml: one visual mesh (what the tactile
camera renders) + a CoACD convex decomposition (what the physics collides with), both in the
same body frame, lying on the table in a resting pose.

    python stl_to_mjcf.py ForkMilimeter.stl --name fork_stl --units mm --threshold 0.01 \
        --face 21.6 0 --site palm 4.5 0 --site neck -2 0 --site handle -45 0 --tine-root 7.05

writes meshes/<name>/:
    <name>_visual.stl      checked/repaired (and, above --max-faces, decimated) mesh, in the STL's units
    <name>_cNN[_tineK].stl convex parts from CoACD, same frame and units
    <name>.xml             MJCF include: <asset> meshes (scale = units -> m) + <body> on the table
    report.json            mesh check, repairs, resting pose, decomposition accuracy

and then you add one line to gelsight_moving.xml, after </worldbody>:
    <include file="meshes/<name>/<name>.xml"/>
(and the name to MESH_TOOLS in gelsight_sim.py; FORKS too if it is a fork).
--active draws the visual mesh in the active group (class mesh_visual_on), so the object is
what you see when gelsight_moving.xml is opened on its own. The convex parts stay
contype 0 in the XML either way: MuJoCo builds a mesh's polygon data (multi-point contacts)
only for meshes that collide at compile time, so compiling them colliding would change the
object's physics (verify_stl.py was run without it). GelSightSim.set_tool() switches them on.

Why two representations: MuJoCo collides a mesh geom through its CONVEX HULL, which for a
fork fills the gaps between the tines. So the plain mesh is visual only (contype 0) and the
collision geometry is a union of convex parts (group 3, invisible) that keeps the gaps open.

Coordinates: all three files keep the STL's own frame (x, y, z in its units); the MJCF scales
them with mesh scale="0.001 0.001 0.001" for millimetres. The body pos/quat put that frame in
the chosen resting pose (a trimesh stable pose; --rest lists them) with the lowest point at
z = 0, turned about the vertical so the STL's +x axis points along world +x.
Sites (--face / --site X Y, STL units): a frame on the upward-facing surface at STL point
(X, Y): +z into the object (along the fitted surface normal), +x = the STL's +x projected onto
the surface. "<name>_face" is the frame GelSightSim.press_relative() uses by default.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001, "um": 1e-6, "in": 0.0254}


def check_mesh(m: trimesh.Trimesh) -> dict:
    edges = m.edges_sorted
    _, cnt = np.unique(edges, axis=0, return_counts=True)
    return dict(triangles=int(len(m.faces)), vertices=int(len(m.vertices)),
                watertight=bool(m.is_watertight), winding_consistent=bool(m.is_winding_consistent),
                is_volume=bool(m.is_volume), bodies=int(len(m.split(only_watertight=False))),
                euler_number=int(m.euler_number),
                degenerate_faces=int((m.area_faces < 1e-12 * max(m.scale, 1e-9) ** 2).sum()),
                boundary_edges=int((cnt == 1).sum()), nonmanifold_edges=int((cnt > 2).sum()),
                bbox_min=m.bounds[0].round(4).tolist(), bbox_max=m.bounds[1].round(4).tolist(),
                extents=m.extents.round(4).tolist(),
                volume=float(m.volume) if m.is_watertight else None, area=float(m.area))


def repair(raw: trimesh.Trimesh, max_faces: int) -> tuple[trimesh.Trimesh, list[str]]:
    changes = []
    m = trimesh.Trimesh(raw.vertices.copy(), raw.faces.copy(), process=True)   # merges duplicate vertices
    if len(m.vertices) != len(raw.vertices):
        changes.append(f"merged duplicate vertices: {len(raw.vertices)} -> {len(m.vertices)} "
                       f"(STL stores 3 per triangle; geometry unchanged)")
    n = len(m.faces)
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    if len(m.faces) != n:
        changes.append(f"removed {n - len(m.faces)} degenerate/duplicate triangles")
    if not m.is_winding_consistent:
        trimesh.repair.fix_winding(m)
        changes.append("fixed inconsistent triangle winding")
    if m.is_watertight and m.volume < 0:
        trimesh.repair.fix_inversion(m)
        changes.append("flipped inside-out normals")
    if not m.is_watertight:
        before = len(m.faces)
        trimesh.repair.fill_holes(m)
        changes.append(f"filled holes: +{len(m.faces) - before} triangles, watertight now {m.is_watertight}")
    if len(m.faces) > max_faces:
        n0 = len(m.faces)
        m, msg = decimate(m, max_faces)
        changes.append(f"decimated {n0} -> {len(m.faces)} triangles ({msg})")
    return m, changes


def _vtk_decimate(m: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
    import vtk
    from vtk.util import numpy_support as ns
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(m.vertices, np.float64), deep=True))
    cells = vtk.vtkCellArray()
    cells.SetData(ns.numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(m.faces) + 1, 3, dtype=np.int64), deep=True),
                  ns.numpy_to_vtkIdTypeArray(np.ascontiguousarray(m.faces, np.int64).ravel(), deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    q = vtk.vtkQuadricDecimation()
    q.SetInputData(pd)
    q.SetTargetReduction(1.0 - target / len(m.faces))
    q.VolumePreservationOn()
    q.Update()
    out = q.GetOutput()
    return trimesh.Trimesh(ns.vtk_to_numpy(out.GetPoints().GetData()),
                           ns.vtk_to_numpy(out.GetPolys().GetConnectivityArray()).reshape(-1, 3), process=True)


def _fqms_decimate(m: trimesh.Trimesh, target: int, agg: float) -> trimesh.Trimesh:
    import fast_simplification
    v, f = fast_simplification.simplify(m.vertices, m.faces, target_count=target, agg=agg)
    return trimesh.Trimesh(v, f, process=True)


def decimate(m: trimesh.Trimesh, max_faces: int) -> tuple[trimesh.Trimesh, str]:
    """Quadric-error decimation: flat regions collapse first, sharp edges such as tine edges
    last. VTK's vtkQuadricDecimation if `vtk` is installed (robust), else fast_simplification
    at decreasing aggressiveness. Every result is checked - triangle count, still watertight and
    in one piece, two-sided deviation from the input - and the first one that passes is used.
    fast_simplification can stall (~200k triangles) or tear thin parts on some meshes; then this
    stops rather than writing a broken mesh."""
    ref = m.sample(20000, seed=0)
    bodies = len(m.split(only_watertight=False))
    attempts = []
    try:
        import vtk  # noqa: F401
        attempts.append(("vtkQuadricDecimation", lambda: _vtk_decimate(m, int(max_faces * 0.99))))
    except ImportError:
        pass
    for agg in (7.0, 4.0, 2.0):          # lower = gentler (keeps shape, may not reach the target)
        attempts.append((f"fast_simplification agg {agg:g}", lambda agg=agg: _fqms_decimate(m, max_faces, agg)))
    tried = []
    for name, run in attempts:
        d = run()
        d.update_faces(d.nondegenerate_faces())
        d.remove_unreferenced_vertices()
        dev = max(trimesh.proximity.closest_point(m, d.sample(20000, seed=1))[1].max(),
                  trimesh.proximity.closest_point(d, ref)[1].max())
        nb = len(d.split(only_watertight=False))
        tried.append(f"{name}: {len(d.faces)} triangles, watertight {d.is_watertight}, {nb} pieces, "
                     f"max deviation {dev:.4g}")
        if len(d.faces) <= max_faces and (d.is_watertight or not m.is_watertight) and nb == bodies:
            return d, f"{name}; two-sided max deviation {dev:.4g} units"
    raise SystemExit("decimation failed every check:\n  " + "\n  ".join(tried) +
                     "\npip install vtk (more robust decimator), decimate the STL elsewhere (MeshLab, "
                     "Blender), or raise --max-faces.")


def rest_poses(m: trimesh.Trimesh):
    """trimesh stable poses, each turned about the vertical so the mesh +x axis points along world +x."""
    T, prob = m.compute_stable_poses(n_samples=1, threshold=0.0)
    out = []
    for t, p in zip(T, prob):
        R = t[:3, :3]
        psi = np.arctan2(R[1, 0], R[0, 0])
        Rz = np.array([[np.cos(-psi), -np.sin(-psi), 0], [np.sin(-psi), np.cos(-psi), 0], [0, 0, 1]])
        R = Rz @ R
        v = m.vertices @ R.T
        low = v[:, 2] < v[:, 2].min() + 1e-6 * m.scale
        out.append(dict(R=R, prob=float(p), z_up_in_mesh=R[2].round(4).tolist(),
                        support_vertices=int(low.sum()),
                        support_mesh_bbox=[m.vertices[low].min(0).round(3).tolist(),
                                           m.vertices[low].max(0).round(3).tolist()]))
    return out


def mat2quat(R):
    import mujoco
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def top_surface_site(m, R, pos, X, Y, radius):
    """Site frame on the upward-facing surface near mesh point (X, Y) (mesh units), in the BODY
    frame: least-squares plane through the top surface within `radius` (gaps are skipped)."""
    zc = m.bounds.mean(0)[2]
    pw = R @ np.array([X, Y, zc]) + pos
    g = np.linspace(-radius, radius, 41)
    gx, gy = np.meshgrid(g, g)
    sel = gx ** 2 + gy ** 2 <= radius ** 2
    O = np.stack([gx[sel] + pw[0], gy[sel] + pw[1], np.full(sel.sum(), pos[2] + 10 * m.scale)], 1)
    D = np.tile([0.0, 0.0, -1.0], (len(O), 1))
    Rinv = R.T
    loc, idx, _ = m.ray.intersects_location((O - pos) @ Rinv.T, D @ Rinv.T, multiple_hits=False)
    if len(idx) < 10:
        raise ValueError(f"site at ({X}, {Y}): only {len(idx)} surface hits within radius {radius}")
    P = loc @ R.T + pos                                   # world hits
    A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
    a, b, c = np.linalg.lstsq(A, P[:, 2], rcond=None)[0]
    p = np.array([pw[0], pw[1], a * pw[0] + b * pw[1] + c])
    n_up = np.array([-a, -b, 1.0]) / np.linalg.norm([-a, -b, 1.0])
    z = -n_up
    e = R[:, 0] - np.dot(R[:, 0], n_up) * n_up
    x = e / np.linalg.norm(e)
    Rw = np.stack([x, np.cross(z, x), z], 1)
    resid = float(np.abs(A @ [a, b, c] - P[:, 2]).max())
    return Rinv @ (p - pos), Rinv @ Rw, dict(hits=int(len(idx)), plane_fit_max_residual=resid,
                                              tilt_deg=float(np.degrees(np.arccos(n_up[2]))))


def decompose(m, threshold, seed=0):
    import coacd
    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(m.vertices, m.faces), threshold=threshold,
                            preprocess_mode="off" if m.is_watertight else "auto",
                            max_convex_hull=-1, resolution=2000, mcts_iterations=150, merge=True,
                            decimate=False, seed=seed)
    hulls = [trimesh.convex.convex_hull(v) for v, _ in parts]
    c = np.array([h.bounds.mean(0) for h in hulls])
    order = np.lexsort((c[:, 1], c[:, 0]))               # sorted along x, then y: easier to read
    return [hulls[i] for i in order]


def hull_deviation(m, hulls, n=4000):
    """Max distance of a collision part's surface OUTSIDE the visual mesh (mesh units)."""
    worst = []
    for h in hulls:
        p = np.vstack([h.sample(max(200, min(n, int(n * h.area / m.area * 20))), seed=1), h.vertices])
        worst.append(float(max(0.0, -trimesh.proximity.signed_distance(m, p).min())))
    return worst


def label_tines(m, hulls, root):
    """Parts that lie entirely beyond x = root (where the slots between the tines end) belong to
    one tine: find which (tines = connected pieces of the mesh just beyond the root, by y)."""
    faces = np.nonzero(m.triangles_center[:, 0] > root + 1e-3 * m.scale)[0]
    beyond = m.submesh([faces], append=True)
    tines = sorted(beyond.split(only_watertight=False), key=lambda t: t.centroid[1])
    labels = []
    for h in hulls:
        if h.bounds[0, 0] < root - 1e-6 * m.scale:
            labels.append(None)
            continue
        # hull vertices lie on the surface of the tine they came from: majority vote
        d = np.stack([trimesh.proximity.closest_point(t, h.vertices)[1] for t in tines], 1)
        labels.append(int(np.bincount(d.argmin(1), minlength=len(tines)).argmax()))
    return labels, len(tines)


def fmt(v):
    v = np.atleast_1d(np.asarray(v, float))
    return " ".join(f"{x:.9g}" for x in np.where(np.abs(v) < 1e-12, 0.0, v))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stl")
    ap.add_argument("--name", required=True, help="body name (geoms/meshes/sites are prefixed with it)")
    ap.add_argument("--units", default="mm", choices=list(UNITS))
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="CoACD concavity threshold, relative to the mesh size (scale-free)")
    ap.add_argument("--max-faces", type=int, default=100_000)
    ap.add_argument("--rest", type=int, default=0, help="index of the stable pose (0 = most likely)")
    ap.add_argument("--list-rest", action="store_true", help="print the stable poses and exit")
    ap.add_argument("--face", type=float, nargs=2, metavar=("X", "Y"), default=None,
                    help="press_relative() frame, STL units (default: centre of the bounding box)")
    ap.add_argument("--site", nargs=3, action="append", default=[], metavar=("NAME", "X", "Y"),
                    help="extra surface frame <name>_<NAME> (STL units)")
    ap.add_argument("--site-radius", type=float, default=2.5, help="surface patch for the site plane fit")
    ap.add_argument("--tine-root", type=float, default=None,
                    help="forks: STL x where the tines start; parts beyond it are named _tineK")
    ap.add_argument("--density", type=float, default=7850.0, help="kg/m^3 (steel)")
    ap.add_argument("--material", default="steel")
    ap.add_argument("--active", action="store_true",
                    help="active object of the raw XML (what MuJoCo shows when the XML is opened on its own)")
    ap.add_argument("--out", default=None, help="default: meshes/<name>/")
    args = ap.parse_args()

    out = args.out or os.path.join(HERE, "meshes", args.name)
    rel = os.path.relpath(out, HERE)
    s = UNITS[args.units]
    raw = trimesh.load(args.stl, process=False, force="mesh")
    report = dict(source=os.path.basename(args.stl), units=args.units, raw=check_mesh(
        trimesh.Trimesh(raw.vertices, raw.faces, process=True)))
    report["raw"]["triangles_in_file"] = int(len(raw.faces))
    m, changes = repair(raw, args.max_faces)
    report["changes"] = changes or ["none"]
    report["final"] = check_mesh(m)
    if not m.is_watertight:
        print("WARNING: mesh is not watertight after repair; volume/inertia/rest pose are unreliable", file=sys.stderr)

    poses = rest_poses(m)
    if args.list_rest:
        for k, p in enumerate(poses):
            print(k, {key: v for key, v in p.items() if key != "R"})
        return
    R = poses[args.rest]["R"]
    v = m.vertices @ R.T
    pos = np.array([0.0, 0.0, -v[:, 2].min()])        # lowest point on the table, STL origin above the table origin

    os.makedirs(out, exist_ok=True)
    hulls = decompose(m, args.threshold)
    # the parts' convex hulls must not reach below the visual mesh's lowest point
    low = min(float((h.vertices @ R.T)[:, 2].min()) for h in hulls)
    if low < -pos[2]:
        pos[2] = -low
    labels, n_tines = (label_tines(m, hulls, args.tine_root) if args.tine_root is not None
                       else ([None] * len(hulls), 0))
    dev = hull_deviation(m, hulls)

    names = []
    for i, (h, lab) in enumerate(zip(hulls, labels)):
        nm = f"{args.name}_c{i:02d}" + (f"_tine{lab}" if lab is not None else "")
        h.export(os.path.join(out, nm + ".stl"))
        names.append(nm)
    m.export(os.path.join(out, f"{args.name}_visual.stl"))

    # inertia of the solid (visual mesh), SI units, mesh frame
    mass = float(m.volume) * s ** 3 * args.density
    com = m.center_mass * s
    I = m.moment_inertia * args.density * s ** 5      # about the centre of mass, mesh axes
    sites = [("face", args.face if args.face is not None else m.bounds.mean(0)[:2])]
    sites += [(n, (float(x), float(y))) for n, x, y in args.site]
    site_xml, site_info = [], {}
    for n, (X, Y) in sites:
        p, Rs, info = top_surface_site(m, R, pos, X, Y, args.site_radius)
        site_info[n] = dict(stl_xy=[X, Y], **info)
        site_xml.append(f'      <site name="{args.name}_{n}" pos="{fmt(p * s)}" quat="{fmt(mat2quat(Rs))}" '
                        f'size="0.001" group="4"/>')

    q = mat2quat(R)
    sc = fmt([s] * 3)
    on = "_on" if args.active else ""
    lines = [f"<!-- generated by stl_to_mjcf.py from {os.path.basename(args.stl)} ({args.units}); do not edit by hand,",
             f"     re-run: python {' '.join(os.path.basename(a) if a == args.stl else a for a in sys.argv)}",
             "     visual mesh: rendered by the tactile camera when active, never collides (class mesh_visual)",
             f"     {len(hulls)} convex parts (CoACD threshold {args.threshold}): collide when active, never drawn",
             "     (class mesh_collision, group 3). All share the body frame = the STL frame."]
    if args.active:
        lines += ["     --active: drawn as the XML's active object (class mesh_visual_on); the parts still",
                  "     collide only once GelSightSim.set_tool() activates them."]
    lines[-1] += " -->"
    lines += ["<mujoco>", "  <asset>",
             f'    <mesh name="{args.name}_visual" file="{rel}/{args.name}_visual.stl" scale="{sc}"/>']
    lines += [f'    <mesh name="{nm}" file="{rel}/{nm}.stl" scale="{sc}"/>' for nm in names]
    lines += ["  </asset>", "  <worldbody>",
              f'    <body name="{args.name}" pos="{fmt(pos * s)}" quat="{fmt(q)}" childclass="tool">',
              f'      <inertial pos="{fmt(com)}" mass="{mass:.9g}" '
              f'fullinertia="{fmt([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])}"/>']
    lines += site_xml
    lines += [f'      <geom name="{args.name}_visual" class="mesh_visual{on}" mesh="{args.name}_visual" '
              f'material="{args.material}"/>']
    lines += [f'      <geom name="{nm}" class="mesh_collision" mesh="{nm}"/>' for nm in names]
    lines += ["    </body>", "  </worldbody>", "</mujoco>", ""]
    with open(os.path.join(out, f"{args.name}.xml"), "w") as f:
        f.write("\n".join(lines))

    report["rest_pose"] = dict(index=args.rest, **{k: v for k, v in poses[args.rest].items() if k != "R"},
                               body_pos_m=(pos * s).tolist(), body_quat=q.tolist(),
                               all=[{k: v for k, v in p.items() if k != "R"} for p in poses[:6]])
    report["inertia"] = dict(mass_kg=mass, com_m=com.tolist(), density=args.density)
    report["decomposition"] = dict(threshold=args.threshold, parts=len(hulls),
                                   hull_vertices=[int(len(h.vertices)) for h in hulls],
                                   max_outside_visual=max(dev), per_part_outside_visual=dev,
                                   tines_found=n_tines, tine_labels=labels)
    report["sites"] = site_info
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("raw", "changes", "final")}, indent=2))
    print(f"rest pose {args.rest}: p={poses[args.rest]['prob']:.2f}, {poses[args.rest]['support_vertices']} support "
          f"vertices, body z {pos[2] * s * 1000:.4f} mm")
    print(f"{len(hulls)} convex parts, max {max(dev):.4f} {args.units} outside the visual mesh; "
          f"{sum(l is not None for l in labels)} parts on {n_tines} tines")
    print(f"wrote {rel}/ ({args.name}.xml, {len(names) + 1} STL files, report.json)")


if __name__ == "__main__":
    main()
