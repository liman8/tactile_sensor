"""
Headless checks of the STL fork (body "fork_stl", from ForkMilimeter.stl) in moving_sensor mode.

    MUJOCO_GL=egl python verify_stl.py        # figures + figures_stl/verify_stl_report.json

  rest       lowest visual / collision point on the table (no penetration, no floating), the
             support points, centre of mass inside the support polygon; MuJoCo's compiled
             visual and collision vertices line up with the STL in one body frame
  a overlay  visual mesh vs CoACD parts: renders (overlay.png), height maps of the gel-facing
             and table-facing sides from orthographic depth renders (deviation.png), cross
             sections (sections.png), widths of the gaps between the tines
  b gaps     first contact between the tines: sphere probes and the gel edge lowered into the
             slots - MuJoCo with the CoACD parts vs exact (visual mesh) vs the convex hull of
             the plain mesh (what MuJoCo would do without the decomposition)
  c footprint 20 random presses 0.5-5 N: region pressed by the physics geometry vs the depth
             render of the visual mesh (IoU, offsets); MuJoCo contact points vs that region
  d sanity   force sweep (touch vs command, depth, settling, jitter), steps/s before/after
             adding the fork, freejoint version
  e gallery  12 tactile images + depth maps across the fork (gallery.png)
  f SDF      MuJoCo's native mesh SDF (octree) instead of CoACD: accuracy and speed
  g detector the existing fork detectors on fork_stl presses
  h regression  every original object gives bit-identical results with and without fork_stl
Exit code 1 if a check fails.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"          # must precede `import mujoco`

import cv2
import mujoco
import numpy as np
import trimesh

from gelsight_sim import GelSightSim, TOOLS, FORKS, COLLISION_GROUP, ACTIVE_GROUP

HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "fork_stl"                     # --name: any object made by stl_to_mjcf.py
MESH_DIR = os.path.join(HERE, "meshes", NAME)
SITES = ["face", "tips", "palm", "neck", "handle", "handle_end"]     # set from the model in main()
XML = None                            # --xml: model to test (default gelsight_moving.xml)
REPORT, FAIL = {}, []


def new_sim(**kw):
    return GelSightSim(xml_path=XML, mode="moving_sensor", **kw)


def object_sites(sim):
    """The object's surface frames: "face" first, then its other <name>_* sites."""
    m = sim.model
    b = sim.tool_body[NAME]
    names = [m.site(i).name[len(NAME) + 1:] for i in range(m.nsite)
             if m.site_bodyid[i] == b and m.site(i).name.startswith(NAME + "_")]
    return ["face"] + [n for n in names if n != "face"]


def is_fork_stl():
    """Fork checks (gaps between the tines, curated gallery) need tine-labelled parts."""
    return any("_tine" in os.path.basename(p) for p in glob.glob(os.path.join(MESH_DIR, f"{NAME}_c*.stl")))


def world_bbox_mm(sim):
    v = to_world_mm(sim, visual_mesh().vertices)
    return v.min(0), v.max(0)


def check(ok, msg):
    if not ok:
        FAIL.append(msg)
        print("  FAIL:", msg)
    return ok


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- model variants (MjSpec edits)
def add_views(spec):
    """Orthographic cameras looking straight down / up, for height maps."""
    for name, z, quat in (("vtop", 0.05, [1, 0, 0, 0]), ("vbot", -0.05, [0, 1, 0, 0])):
        c = spec.worldbody.add_camera()
        c.name, c.pos, c.quat = name, [0, 0, z], quat
        c.proj = mujoco.mjtProjection.mjPROJ_ORTHOGRAPHIC
        c.fovy = 0.018


def add_probe(spec):
    """Sphere indenter (only used kinematically, with mj_geomDistance)."""
    b = spec.worldbody.add_body()
    b.name, b.pos = "probe", [0, 0, 0.5]
    b.add_freejoint()
    g = b.add_geom()
    g.name, g.type, g.size = "probe", mujoco.mjtGeom.mjGEOM_SPHERE, [0.001, 0, 0]
    g.contype = g.conaffinity = 0
    g.group = 4


def remove_object(spec):
    """The model as it was before the STL object was added."""
    spec.delete(spec.body(NAME))
    for m in list(spec.meshes):
        if m.name.startswith(NAME):
            spec.delete(m)


def sdf_variant(depth=None):
    """fork_stl's 17 convex parts replaced by one geom type="sdf" on the visual mesh (MuJoCo
    builds a signed distance field octree from it; SDF meshes cannot be scaled, so the
    vertices are given in metres)."""
    vis = trimesh.load(os.path.join(MESH_DIR, f"{NAME}_visual.stl"))

    def edit(spec):
        parts = [g for g in spec.body(NAME).geoms if g.name.startswith(f"{NAME}_c")]
        for g in parts[1:]:
            spec.delete(g)
        mesh = spec.add_mesh()
        mesh.name = f"{NAME}_sdf"
        mesh.uservert = (vis.vertices * 0.001).ravel().tolist()
        mesh.userface = vis.faces.ravel().tolist()
        mesh.needsdf = 1
        if depth:
            mesh.octree_maxdepth = depth
        g = parts[0]                      # keeps the parts' class: group 3, contact parameters
        g.name, g.type, g.meshname = f"{NAME}_sdf", mujoco.mjtGeom.mjGEOM_SDF, f"{NAME}_sdf"
    return edit


def edits(*fs):
    def run(spec):
        for f in fs:
            f(spec)
    return run


# ---------------------------------------------------------------- geometry helpers
def visual_mesh():
    return trimesh.load(os.path.join(MESH_DIR, f"{NAME}_visual.stl"))


def part_meshes():
    return [trimesh.load(p) for p in sorted(glob.glob(os.path.join(MESH_DIR, f"{NAME}_c*.stl")))]


def body_pose(sim):
    b = sim.tool_body[NAME]
    return sim.data.xmat[b].reshape(3, 3).copy(), sim.data.xpos[b].copy()


def to_world_mm(sim, pts_mm):
    R, p = body_pose(sim)
    return np.asarray(pts_mm) @ R.T + p * 1000.0


def geom_world_vertices(sim, g):
    m, d = sim.model, sim.data
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


def obj_parts(sim):
    """Collision parts of the object under test (other mesh objects have theirs too)."""
    return sorted(g for g in sim.collision_geoms if sim.model.geom_bodyid[g] == sim.tool_body[NAME])


def obj_visual(sim):
    return [g for g in sim.visual_geoms if sim.model.geom_bodyid[g] == sim.tool_body[NAME]]


class show_parts:
    """While rendering group 3: draw this object's collision parts (alpha 1) and move every
    other object's parts out of the way (group 4)."""

    def __init__(self, sim):
        self.m, self.mine = sim.model, obj_parts(sim)
        self.others = sorted(set(sim.collision_geoms) - set(self.mine))

    def __enter__(self):
        self.rgba, self.group = self.m.geom_rgba.copy(), self.m.geom_group.copy()
        self.m.geom_rgba[self.mine, 3] = 1.0
        self.m.geom_group[self.others] = 4
        return self

    def __exit__(self, *exc):
        self.m.geom_rgba[:], self.m.geom_group[:] = self.rgba, self.group


def indentation_from_group(sim, group):
    """Like GelSightSim.indentation_mm(raw=True), but rendering geom group `group` (3 = the
    collision parts) through the tactile camera."""
    m = sim.model
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[group] = 1
    opt.sitegroup[:] = 0
    with show_parts(sim):
        sim._renderer.update_scene(sim.data, camera=sim.cam_id, scene_option=opt)
        depth = sim._renderer.render().astype(np.float32)
    depth = cv2.medianBlur(depth, 3).astype(np.float64)
    return np.clip((sim.d0 - depth) * 1000.0, 0.0, sim.opt.max_indent_mm)


class HeightMaps:
    """Orthographic depth renders of one geom group, straight down ("top", the gel-facing side
    of the resting fork) or straight up from under the table ("bot"), stitched along x.
    Returns heights [mm] on a world grid (nan = nothing there)."""

    def __init__(self, sim, s_mm=0.04, H=450, W=1000):
        self.sim, self.s, self.H, self.W = sim, s_mm, H, W
        m = sim.model
        self.r = mujoco.Renderer(m, H, W)
        self.r.enable_depth_rendering()
        self.cam = {"top": m.camera("vtop").id, "bot": m.camera("vbot").id}
        for c in self.cam.values():
            m.cam_fovy[c] = H * s_mm / 1000.0

    def grid(self, x0, ntiles, y0=0.0):
        xs = x0 + np.arange(ntiles * self.W) * self.s
        ys = y0 + ((self.H - 1) / 2 - np.arange(self.H)) * self.s          # row 0 = +y
        return xs, ys

    def render(self, side, groups, x0, ntiles, y0=0.0):
        sim, m, d = self.sim, self.sim.model, self.sim.data
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        for g in groups:
            opt.geomgroup[g] = 1
        opt.sitegroup[:] = 0
        parts = show_parts(sim).__enter__()
        c = self.cam[side]
        z = 0.05 if side == "top" else -0.05
        tiles = []
        for k in range(ntiles):
            cx = (x0 + (k * self.W + (self.W - 1) / 2) * self.s) / 1000.0
            m.cam_pos[c] = (cx, y0 / 1000.0, z)
            mujoco.mj_camlight(m, d)
            self.r.update_scene(d, camera=c, scene_option=opt)
            # 3x3 median, as in GelSightSim.indentation_mm: multisampled silhouette pixels
            # resolve to a mix of object and background depth
            dep = cv2.medianBlur(self.r.render().astype(np.float32), 3).astype(np.float64)
            if side == "bot":
                dep = dep[::-1]                        # camera turned about x: image rows flipped
            h = (z - dep) * 1000.0 if side == "top" else (z + dep) * 1000.0
            h[dep > 1.0] = np.nan
            tiles.append(h)
        parts.__exit__()
        return np.concatenate(tiles, 1)


def gaps_from_intervals(iv, lo, hi):
    """Gaps (y ranges with no material) between merged intervals, inside [lo, hi]."""
    iv = sorted(iv)
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)
            if merged[i][1] > lo and merged[i + 1][0] < hi]


def section_intervals(mesh, x):
    sec = mesh.section(plane_origin=[x, 0, 0], plane_normal=[1, 0, 0])
    if sec is None:
        return [], []
    polys = [p[:, [1, 2]] for p in sec.discrete]
    return [(p[:, 0].min(), p[:, 0].max()) for p in polys], polys


# ---------------------------------------------------------------- 0. rest
def check_rest(sim):
    log("\n== rest pose ==")
    m, d = sim.model, sim.data
    out = {}
    vis_w = to_world_mm(sim, visual_mesh().vertices)
    for placement in [dict(x=0.0, y=0.0, yaw=0.0), dict(x=0.012, y=-0.02, yaw=1.1), dict(x=-0.03, y=0.01, yaw=-2.5)]:
        sim.place_object(NAME, **placement)
        lv = min(geom_world_vertices(sim, g)[:, 2].min() for g in obj_visual(sim))
        lc = min(geom_world_vertices(sim, g)[:, 2].min() for g in obj_parts(sim))
        out[str(placement)] = dict(lowest_visual_um=lv * 1e6, lowest_collision_um=lc * 1e6)
        log(f"  placed at {placement}: lowest visual vertex {lv * 1e6:+.6f} um, lowest collision vertex {lc * 1e6:+.6f} um")
        check(-1e-3 < lv * 1e6 < 1e-3 and -1e-3 < lc * 1e6 < 1e-3, f"rest {placement}: lowest point not at z=0")
    sim.place_object(NAME)
    # compiled geometry vs the STL: visual and collision line up in one body frame
    from scipy.spatial import cKDTree
    R, p = body_pose(sim)
    vis = visual_mesh()

    def mismatch(a, b):                         # symmetric nearest-vertex distance [mm]
        return max(cKDTree(b).query(a)[0].max(), cKDTree(a).query(b)[0].max())
    gv = geom_world_vertices(sim, m.geom(f"{NAME}_visual").id) * 1000.0
    err_v = mismatch(gv, vis.vertices @ R.T + p * 1000.0)
    err_c = 0.0
    for g, pm in zip(obj_parts(sim), part_meshes()):
        gw = geom_world_vertices(sim, g) * 1000.0
        err_c = max(err_c, mismatch(gw, pm.vertices @ R.T + p * 1000.0))
    out["compiled_vs_stl_mm"] = dict(visual=float(err_v), collision=float(err_c))
    log(f"  MuJoCo's compiled vertices vs the STL files (same body frame): visual {err_v:.2e} mm, parts {err_c:.2e} mm")
    check(err_v < 1e-4 and err_c < 1e-4, "compiled meshes do not line up with the STL")    # float32 storage
    # support points and centre of mass
    sup = vis.vertices[vis_w[:, 2] < 1e-3]            # within 1 um of the table
    com = d.xipos[sim.tool_body[NAME]] * 1000.0
    sw = (sup @ R.T + p * 1000.0)[:, :2].astype(np.float32)
    hull = cv2.convexHull(sw)
    inside = cv2.pointPolygonTest(hull, (float(com[0]), float(com[1])), True)
    out["support"] = dict(n=int(len(sup)), stl_x=sorted(set(np.round(sup[:, 0], 2).tolist())),
                          com_mm=com.round(3).tolist(), com_inside_margin_mm=float(inside))
    log(f"  {len(sup)} support vertices at STL x = {out['support']['stl_x']}"
        f"{' (tine tips + handle end)' if NAME == 'fork_stl' else ''}; centre of mass {inside:.2f} mm inside the "
        f"support polygon")
    check(inside > 0, "centre of mass outside the support polygon")
    if NAME == "fork_stl":
        check(sup[:, 0].min() < -80 and sup[:, 0].max() > 36, "fork does not rest on tine tips and handle end")

    # freejoint version: let it lie for 2 s
    sf = new_sim(free_objects=True)
    sf.place_object(NAME)
    c0 = sf.object_com()
    lowest = []
    for k in range(4000):
        mujoco.mj_step(sf.model, sf.data)
        if k % 100 == 99:
            lowest.append(min(geom_world_vertices(sf, g)[:, 2].min() for g in obj_parts(sf)))
    v = np.empty(6)
    mujoco.mj_objectVelocity(sf.model, sf.data, mujoco.mjtObj.mjOBJ_BODY, sf.tool_body[NAME], v, 0)
    shift = np.linalg.norm(sf.object_com() - c0) * 1000.0
    out["free_drop_2s"] = dict(com_shift_mm=float(shift), speed_mm_s=float(np.linalg.norm(v[3:]) * 1000),
                               ang_speed_rad_s=float(np.linalg.norm(v[:3])),
                               lowest_collision_um=float(min(lowest) * 1e6), lowest_final_um=float(lowest[-1] * 1e6))
    log(f"  freejoint, 2 s on the table: moved {shift * 1000:.1f} um, speed {np.linalg.norm(v[3:]) * 1e6:.2f} um/s, "
        f"deepest point {min(lowest) * 1e6:+.2f} um (soft contact), final {lowest[-1] * 1e6:+.2f} um")
    check(shift < 0.05 and abs(min(lowest)) < 20e-6, "freejoint fork does not rest still on the table")
    REPORT["rest"] = out


# ---------------------------------------------------------------- a. overlay
def check_overlay(sim, out_dir):
    log("\n== a. overlay: visual mesh vs CoACD collision parts ==")
    m, d = sim.model, sim.data
    sim.place_object(NAME)
    rep = {}
    lo, hi = world_bbox_mm(sim)
    hm = HeightMaps(sim, H=int(min(950, np.ceil((hi[1] - lo[1] + 2.0) / 0.04))))
    x0 = float(np.floor(lo[0] - 2.0))
    nt = int(np.ceil((hi[0] + 2.0 - x0) / (hm.W * hm.s)))
    y0 = float(0.5 * (lo[1] + hi[1]))
    xs, ys = hm.grid(x0, nt, y0)
    maps = {}
    for side in ("top", "bot"):
        maps[side] = (hm.render(side, [ACTIVE_GROUP], x0, nt, y0), hm.render(side, [COLLISION_GROUP], x0, nt, y0))
    px = hm.s ** 2
    for side, label in (("top", "gel side (up)"), ("bot", "table side (down)")):
        hv, hc = maps[side]
        # compare away from the silhouettes (a pixel on an outline sees a side wall in one
        # render and a top face in the other)
        k = np.ones((3, 3), np.uint8)
        both = (cv2.erode((~np.isnan(hv)).astype(np.uint8), k) & cv2.erode((~np.isnan(hc)).astype(np.uint8), k)).astype(bool)
        dev = (hc - hv) if side == "top" else (hv - hc)            # > 0: collision sticks out
        filled = np.isnan(hv) & ~np.isnan(hc)
        missing = ~np.isnan(hv) & np.isnan(hc)
        # 1-px silhouette rim: two different triangle sets rasterised at the same edge
        rim = cv2.dilate((~np.isnan(hv)).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        r = dict(max_mm=float(np.max(dev[both])), p99_mm=float(np.percentile(dev[both], 99)),
                 mean_mm=float(np.mean(dev[both])), min_mm=float(np.min(dev[both])),
                 filled_mm2=float(filled.sum() * px), filled_beyond_1px_rim_mm2=float((filled & ~rim).sum() * px),
                 missing_mm2=float(missing.sum() * px), area_mm2=float((~np.isnan(hv)).sum() * px))
        rep[side] = r
        log(f"  {label:18s}: collision surface outside the visual by max {r['max_mm']:.3f} mm, p99 {r['p99_mm']:.4f}, "
            f"mean {r['mean_mm']:.4f} (inside by at most {-r['min_mm']:.4f}); area only in collision "
            f"{r['filled_mm2']:.2f} mm2 ({r['filled_beyond_1px_rim_mm2']:.3f} beyond a 1-px rim), "
            f"only in visual {r['missing_mm2']:.2f} mm2, fork area {r['area_mm2']:.0f} mm2")
    top_dev = rep["top"]
    check(top_dev["p99_mm"] < 0.02 and top_dev["filled_beyond_1px_rim_mm2"] < 0.5,
          "gel-side collision surface deviates from the visual mesh")
    # where is the worst deviation (STL coordinates)?
    for side in ("top", "bot"):
        hv, hc = maps[side]
        k = np.ones((3, 3), np.uint8)
        both = (cv2.erode((~np.isnan(hv)).astype(np.uint8), k) & cv2.erode((~np.isnan(hc)).astype(np.uint8), k)).astype(bool)
        dev = np.where(both, (hc - hv) if side == "top" else (hv - hc), -1.0)
        i, j = np.unravel_index(np.argmax(dev), dev.shape)
        R, p = body_pose(sim)
        w = np.array([xs[j], ys[i], hv[i, j]])
        stl = R.T @ (w - p * 1000.0)
        rep[side]["worst_at_stl_mm"] = stl.round(2).tolist()
        log(f"  worst {side} deviation at STL (x, y) = ({stl[0]:.1f}, {stl[1]:.1f}) mm")

    # 3D: farthest point of the collision parts outside the visual mesh, and visual surface not covered
    vis, parts = visual_mesh(), part_meshes()
    out3 = []
    for h in parts:
        pts = np.vstack([h.sample(3000, seed=1), h.vertices])
        out3.append(float(max(0.0, -trimesh.proximity.signed_distance(vis, pts).min())))
    vs = vis.sample(30000, seed=2)
    dmin = np.full(len(vs), np.inf)
    for h in parts:
        dist = trimesh.proximity.closest_point(h, vs)[1]
        dist[h.contains(vs)] = 0.0
        dmin = np.minimum(dmin, dist)
    rep["3d"] = dict(collision_outside_visual_max_mm=max(out3), per_part_mm=out3,
                     visual_not_covered_max_mm=float(dmin.max()))
    log(f"  3D: collision parts reach at most {max(out3):.3f} mm outside the visual mesh (part "
        f"{int(np.argmax(out3))}: the underside chord of a curved tine); every visual surface point is within "
        f"{dmin.max() * 1000:.1f} um of the collision geometry")
    check(dmin.max() < 0.01, "visual surface not covered by the collision parts")

    # gaps between tines: plan-view cross sections, visual vs union of parts
    gap_rows = []
    for x in ([7.5, 8, 9, 10, 12, 15, 20, 25, 30, 34, 36] if is_fork_stl() else []):
        iv_v, _ = section_intervals(vis, x)
        iv_c = sum((section_intervals(h, x)[0] for h in parts), [])
        gv = gaps_from_intervals(iv_v, -9, 9)
        gc = gaps_from_intervals(iv_c, -9, 9)
        gap_rows.append(dict(stl_x=x, visual_gaps_mm=[round(float(b - a), 4) for a, b in gv],
                             collision_gaps_mm=[round(float(b - a), 4) for a, b in gc]))
        log(f"  x = {x:4.1f} mm: gap widths visual {[round(float(b - a), 3) for a, b in gv]}  collision "
            f"{[round(float(b - a), 3) for a, b in gc]}")
        check(len(gc) == len(gv) and all(abs((b - a) - (bb - aa)) < 0.01 for (a, b), (aa, bb) in zip(gv, gc)),
              f"gap between tines filled/narrowed at x = {x} mm")
    rep["gaps"] = gap_rows

    # ---- figures
    _deviation_figure(maps, xs, ys, out_dir, lo, hi)
    R, _ = body_pose(sim)
    _sections_figure(vis, parts, out_dir, up_z=float(R[2, 2]))
    _overlay_render(sim, out_dir)
    REPORT["overlay"] = rep


def _colorbar(h, w, lo, hi, cmap, label):
    bar = np.zeros((h, w, 3), np.uint8)
    g = cv2.applyColorMap(np.linspace(0, 255, w - 120).astype(np.uint8)[None].repeat(14, 0), cmap)
    bar[4:18, 60:w - 60] = g
    cv2.putText(bar, f"{lo:g}", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, f"{hi:g}", (w - 55, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, label, (60, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def _bar(width, text, h=26, color=(255, 255, 255), scale=0.55):
    b = np.zeros((h, width, 3), np.uint8)
    cv2.putText(b, text, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    return b


def _deviation_figure(maps, xs, ys, out_dir, lo, hi):
    """Top: height of the gel-facing surface. Middle/bottom: collision minus visual (gel side /
    table side), red = only collision geometry (gap filled), blue = only visual."""
    sel = (xs > lo[0] - 2) & (xs < hi[0] + 2)
    rows = []
    hv, hc = maps["top"]
    h = np.nan_to_num(hv[:, sel], nan=0)
    img = cv2.applyColorMap((np.clip(h / 4.0, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    img[np.isnan(hv[:, sel])] = 30
    rows += [_bar(img.shape[1], "visual mesh, gel-facing surface height above the table (orthographic depth render, 0.04 mm/px)"),
             img, _colorbar(30, img.shape[1], 0, 4, cv2.COLORMAP_VIRIDIS, "height [mm]")]
    for side, title in (("top", "gel side"), ("bot", "table side")):
        hv, hc = maps[side]
        dev = (hc - hv) if side == "top" else (hv - hc)
        dev = dev[:, sel]
        vmax = 0.05 if side == "top" else 0.5
        img = cv2.applyColorMap((np.clip(np.nan_to_num(dev, nan=0) / vmax, 0, 1) * 255).astype(np.uint8),
                                cv2.COLORMAP_INFERNO)
        nv, nc = np.isnan(hv[:, sel]), np.isnan(hc[:, sel])
        img[nv & nc] = 30
        img[nv & ~nc] = (0, 0, 255)
        img[~nv & nc] = (255, 128, 0)
        rows += [_bar(img.shape[1], f"{title}: collision parts minus visual mesh "
                      f"(red = collision only, i.e. gap filled; blue = visual only)"),
                 img, _colorbar(30, img.shape[1], 0, vmax, cv2.COLORMAP_INFERNO, "deviation [mm]")]
    fig = np.concatenate(rows, 0)
    if not is_fork_stl():
        cv2.imwrite(os.path.join(out_dir, "deviation.png"), fig)
        return
    # zoom on the head (tines) at 2x
    head = (xs[sel] > -6) & (xs[sel] < 38)
    hv, hc = maps["top"]
    dev = (hc - hv)[:, sel][:, head]
    img = cv2.applyColorMap((np.clip(np.nan_to_num(dev, nan=0) / 0.05, 0, 1) * 255).astype(np.uint8),
                            cv2.COLORMAP_INFERNO)
    nv, nc = np.isnan(hv[:, sel][:, head]), np.isnan(hc[:, sel][:, head])
    img[nv & nc] = 30
    img[nv & ~nc] = (0, 0, 255)
    img[~nv & nc] = (255, 128, 0)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    pad = np.zeros((img.shape[0], fig.shape[1] - img.shape[1], 3), np.uint8)
    fig = np.concatenate([fig, _bar(fig.shape[1], "head, gel side, 2x zoom (0.02 mm/px on screen): tine gaps stay open"),
                          np.concatenate([img, pad], 1)], 0)
    cv2.imwrite(os.path.join(out_dir, "deviation.png"), fig)


def _sections_figure(vis, parts, out_dir, up_z, ppm=36):
    """Cross sections (STL y-z, drawn with the gel-facing side up): visual mesh filled grey,
    each CoACD part outlined in colour. up_z: sign of the STL z axis that points up at rest."""
    (bx0, by0, bz0), (bx1, by1, bz1) = vis.bounds
    if is_fork_stl():
        xs = [8, 12, 20, 28, 34, 4.5, -2, -45]
    else:
        xs = list(np.round(np.linspace(bx0, bx1, 10)[1:-1], 2))
    ppm = min(ppm, 680.0 / (by1 - by0 + 3.0))
    W, H = int((by1 - by0 + 3.0) * ppm), int((bz1 - bz0 + 1.5) * ppm)
    colors = [tuple(int(c) for c in cv2.applyColorMap(np.uint8([[i * 255 // max(len(parts) - 1, 1)]]),
                                                      cv2.COLORMAP_TURBO)[0, 0]) for i in range(len(parts))]
    panels = []
    for x in xs:
        img = np.full((H, W, 3), 25, np.uint8)
        to_px = lambda P: np.c_[(P[:, 0] - by0 + 1.5) * ppm,
                                ((P[:, 1] - bz0) if up_z < 0 else (bz1 - P[:, 1])) * ppm + 0.75 * ppm
                                ].round().astype(np.int32)                         # gel-facing side up
        _, polys = section_intervals(vis, x)
        for P in polys:
            cv2.fillPoly(img, [to_px(P)], (150, 150, 150), cv2.LINE_AA)
        for k, h in enumerate(parts):
            _, pp = section_intervals(h, x)
            for P in pp:
                cv2.polylines(img, [to_px(P)], True, colors[k], 2, cv2.LINE_AA)

        cv2.putText(img, f"STL x = {x:g} mm", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "gel side", (W - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        panels.append(cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(70, 70, 70)))
    grid = np.concatenate([np.concatenate(panels[i:i + 2], 1) for i in range(0, len(panels), 2)], 0)
    title = _bar(grid.shape[1], f"cross sections: grey = visual mesh, coloured outlines = the {len(parts)} CoACD parts "
                 f"({ppm:.0f} px/mm, gel-facing side up)")
    cv2.imwrite(os.path.join(out_dir, "sections.png"), np.concatenate([title, grid], 0))


def _overlay_render(sim, out_dir):
    """Renders of the collision parts (one colour each) with the visual mesh's outline drawn on
    top (white, from a segmentation render of the visual geom alone). Pixels where the two
    silhouettes disagree are magenta."""
    m, d = sim.model, sim.data
    W, H = 900, 520
    r = mujoco.Renderer(m, H, W)
    rgba = m.geom_rgba.copy()
    cols = obj_parts(sim)
    others = sorted(set(sim.collision_geoms) - set(cols))
    group = m.geom_group.copy()
    m.geom_group[others] = 4
    for k, g in enumerate(cols):
        c = cv2.applyColorMap(np.uint8([[k * 255 // max(len(cols) - 1, 1)]]), cv2.COLORMAP_TURBO)[0, 0][::-1] / 255
        m.geom_rgba[g] = (*c, 1.0)
    vis_g = m.geom(f"{NAME}_visual").id
    cam = mujoco.MjvCamera()
    lo, hi = world_bbox_mm(sim)
    ctr, size = (lo + hi) / 2000.0, float((hi - lo).max()) / 1000.0

    def render(cv, groups, seg=False):
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        for g in groups:
            opt.geomgroup[g] = 1
        opt.sitegroup[:] = 0
        cam.lookat[:] = cv["lookat"]
        cam.distance, cam.azimuth, cam.elevation = cv["distance"], cv["azimuth"], cv["elevation"]
        if seg:
            r.enable_segmentation_rendering()
        r.update_scene(d, camera=cam, scene_option=opt)
        img = r.render()
        if seg:
            r.disable_segmentation_rendering()
            return img[..., 0]
        return img[..., ::-1].copy()

    def overlay_shot(title, cv):
        img = render(cv, [COLLISION_GROUP])
        mv = render(cv, [ACTIVE_GROUP], seg=True) == vis_g
        mc = np.isin(render(cv, [COLLISION_GROUP], seg=True), cols)
        edge = cv2.morphologyEx(mv.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        img[edge] = 255
        img[mv ^ mc] = (255, 0, 255)
        return np.concatenate([_bar(W, f"{title} | silhouettes differ in {int((mv ^ mc).sum())} of "
                                       f"{int(mv.sum())} px", scale=0.47), img], 0)

    def plain_shot(title, cv, groups):
        return np.concatenate([_bar(W, title, scale=0.47), render(cv, groups)], 0)

    if is_fork_stl():
        top = dict(lookat=(0.0145, 0, 0.003), distance=0.062, azimuth=-90, elevation=-89.9)
        obl = dict(lookat=(0.012, 0, 0.003), distance=0.075, azimuth=-130, elevation=-40)
        side = dict(lookat=(0.022, 0, 0.002), distance=0.045, azimuth=-90, elevation=-1)
        whole = dict(lookat=(-0.024, 0, 0.002), distance=0.17, azimuth=-90, elevation=-89.9)
    else:
        top = whole = dict(lookat=tuple(ctr), distance=1.2 * size, azimuth=-90, elevation=-89.9)
        obl = dict(lookat=tuple(ctr), distance=1.1 * size, azimuth=-130, elevation=-40)
        side = dict(lookat=tuple(ctr), distance=1.0 * size, azimuth=-90, elevation=-1)
    shots = [overlay_shot("from above: CoACD parts (colours) + visual mesh outline (white)", top),
             overlay_shot("whole object from above, same colours", whole),
             plain_shot("collision parts only (with the table)", obl, [0, COLLISION_GROUP]),
             plain_shot("visual mesh only, same view", obl, [0, ACTIVE_GROUP]),
             overlay_shot("side view: parts + visual outline (hull bulges below curved tines)", side)]
    m.geom_rgba[:] = rgba
    m.geom_group[:] = group
    r.close()
    legend = np.zeros_like(shots[-1])
    for k, line in enumerate(["white: outline of the visual mesh (segmentation render)",
                              "magenta: pixels covered by only one of visual / collision",
                              "colours: one per CoACD part (geom group 3, never drawn",
                              "         by the tactile camera, rgba alpha 0 in the model)",
                              "top views: shared silhouette; side view: the hull chord",
                              "under each curved tine (table side, <= 0.45 mm)"]):
        cv2.putText(legend, line, (20, 60 + 32 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    shots.append(legend)
    fig = np.concatenate([np.concatenate(shots[i:i + 2], 1) for i in range(0, 6, 2)], 0)
    cv2.imwrite(os.path.join(out_dir, "overlay.png"), fig)


# ---------------------------------------------------------------- b. between the tines
def first_contact(z_hi, z_lo, touching, step=0.05, iters=30):
    """Lower something from z_hi to z_lo [mm] and return the height of the first contact
    (scan in `step`s, then bisection), or None if it gets down to z_lo without touching.
    A scan, not a plain bisection: a probe can slip under a raised part without touching."""
    zs = np.arange(z_hi, z_lo - 1e-9, -step)
    prev = z_hi
    for z in zs:
        if touching(z):
            lo, hi = z, prev
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                if touching(mid):
                    lo = mid
                else:
                    hi = mid
            return lo
        prev = z
    return None


def check_gaps(out_dir, variants):
    """variants: {label: GelSightSim with a probe body}. Sphere probes lowered between the tines
    (and the gel edge), first-contact height: exact (visual mesh) vs each variant vs the plain
    mesh's convex hull."""
    log("\n== b. between the tines: first contact ==")
    rep = dict(probes=[], gel_edge=[])
    sim0 = next(iter(variants.values()))
    for s in variants.values():
        s.place_object(NAME)
    vis = visual_mesh()
    R, p = body_pose(sim0)
    vis_w = trimesh.Trimesh(vis.vertices @ R.T + p * 1000.0, vis.faces)     # world, mm
    centre_y = {"centre": 0.0}
    # gap centre lines in STL y: side gaps between inner and outer tines (both sides)
    for x in (9.0,):
        g = gaps_from_intervals(section_intervals(vis, x)[0], -9, 9)
    side = [0.5 * (a + b) for a, b in g if abs(0.5 * (a + b)) > 1]
    centre_y.update({"side+": max(side), "side-": min(side)})

    def exact(c_w, r):
        def touching(z):
            q = np.array([[c_w[0], c_w[1], z]])
            dist = trimesh.proximity.closest_point(vis_w, q)[1][0]
            return dist <= r or bool(vis_w.contains(q)[0])
        return touching

    def via_geoms(sim, geoms, c_w, r):
        m, d = sim.model, sim.data
        pg = m.geom("probe").id
        m.geom_size[pg, 0] = r / 1000.0
        m.geom_rbound[pg] = r / 1000.0
        qa = m.joint(m.body("probe").jntadr[0]).qposadr[0]

        def touching(z):
            d.qpos[qa:qa + 3] = (c_w[0] / 1000, c_w[1] / 1000, z / 1000)
            d.qpos[qa + 3:qa + 7] = (1, 0, 0, 0)
            mujoco.mj_kinematics(m, d)
            return any(mujoco.mj_geomDistance(m, d, pg, g, 0.002, None) < 0 for g in geoms)
        return touching

    rows = []
    for gap, y_stl in centre_y.items():
        for x_stl in (9.0, 12.0, 16.0, 20.0, 25.0, 30.0, 34.0):
            iv = section_intervals(vis, x_stl)[0]
            gw = [b - a for a, b in gaps_from_intervals(iv, -9, 9) if a < y_stl < b]
            width = gw[0] if gw else 0.0
            # tine top (gel side) height next to the gap
            for r in (0.3, 0.6, 1.0, 1.5):
                c_w = (np.array([x_stl, y_stl, 2.5]) @ R.T + p * 1000.0)
                z_hi, z_lo = 9.0, r                    # probe centre heights [mm]; z_lo = resting on the table
                res = dict(gap=gap, stl_x=x_stl, gap_width_mm=round(width, 3), probe_r_mm=r)
                res["exact"] = first_contact(z_hi, z_lo, exact(c_w, r))
                if res["exact"] is None:          # slips through: by how much?
                    zz = np.arange(z_lo, z_hi, 0.02)
                    q = np.c_[np.full_like(zz, c_w[0]), np.full_like(zz, c_w[1]), zz]
                    res["exact_clearance_mm"] = float(trimesh.proximity.closest_point(vis_w, q)[1].min() - r)
                for label, s in variants.items():
                    res[label] = first_contact(z_hi, z_lo, via_geoms(s, obj_parts(s), c_w, r))
                res["convex_hull_of_mesh"] = first_contact(z_hi, z_lo, via_geoms(
                    sim0, [sim0.model.geom(f"{NAME}_visual").id], c_w, r))
                rows.append(res)
    rep["probes"] = rows
    # summary per variant
    # a probe that slips through the exact geometry with < 20 um to spare is a tie at the
    # decomposition's tolerance (hulls may bulge ~10-20 um sideways): reported, not judged
    tie = [r_.get("exact_clearance_mm", 1.0) < 0.02 for r_ in rows]
    for label in list(variants) + ["convex_hull_of_mesh"]:
        agree, errs, through, wrong = 0, [], 0, []
        for r_, t in zip(rows, tie):
            e, v = r_["exact"], r_[label]
            if e is None and v is None:
                agree += 1
                through += 1
            elif e is not None and v is not None:
                errs.append(v - e)
                agree += abs(v - e) < 0.02
            if not t and not ((e is None and v is None) or (e is not None and v is not None and abs(v - e) < 0.02)):
                wrong.append(r_)
        n = len(rows)
        n_through = sum(r_["exact"] is None for r_ in rows)
        s = dict(n=n, agree_within_20um=agree, passes_through_exact=n_through, passes_through=through,
                 ties_within_20um=int(sum(tie)), disagree_excluding_ties=len(wrong),
                 max_abs_err_mm=float(np.max(np.abs(errs))) if errs else 0.0,
                 mean_err_mm=float(np.mean(errs)) if errs else 0.0)
        rep[f"summary_{label}"] = s
        log(f"  sphere probes, {label:20s}: {agree}/{n} agree with the exact geometry within 20 um "
            f"({through}/{n_through} slip through to the table like the exact geometry); {len(wrong)} disagree "
            f"excluding {int(sum(tie))} ties; max |height error| {s['max_abs_err_mm']:.4f} mm")
    for r_, t in zip(rows, tie):
        if t:
            log(f"    tie: {r_['gap']} gap at x = {r_['stl_x']} mm ({r_['gap_width_mm']:.3f} mm wide), probe r = "
                f"{r_['probe_r_mm']} mm slips through the visual mesh with {r_['exact_clearance_mm'] * 1000:.1f} um to "
                f"spare; " + ", ".join(f"{k}: {'through' if r_[k] is None else f'touches at {r_[k]:.3f} mm'}"
                                       for k in variants))
    check(rep["summary_coacd"]["disagree_excluding_ties"] == 0, "CoACD: probe first contact differs from the exact geometry")
    # a few rows for the log
    for r_ in rows[::7]:
        f = lambda v: "through" if v is None else f"{v:.3f}"
        log(f"    {r_['gap']:6s} x={r_['stl_x']:4.1f} gap {r_['gap_width_mm']:.2f} mm, r={r_['probe_r_mm']}: exact "
            f"{f(r_['exact'])}  " + "  ".join(f"{k} {f(r_[k])}" for k in list(variants) + ["convex_hull_of_mesh"]))

    # ---- gel edge, lowered into the centre gap and a side gap
    vs = np.vstack([vis.sample(400000, seed=3), vis.vertices])
    E = vis.edges_unique
    t = np.linspace(0, 1, 60)[None, :, None]
    vs = np.vstack([vs, (vis.vertices[E[:, 0]][:, None] * (1 - t) + vis.vertices[E[:, 1]][:, None] * t).reshape(-1, 3)])
    pts_w = (vs @ R.T + p * 1000.0) / 1000.0                       # m
    for gap in ("centre", "side+"):
        for dy in (-0.25, 0.0, 0.25):
            row = dict(gap=gap, offset_mm=dy)
            for label, s in list(variants.items()) + [("convex_hull_of_mesh", sim0)]:
                m, d = s.model, s.data
                gid = s.gel_id
                target = (np.array([25.0, centre_y[gap] + dy, 2.5]) @ R.T + p * 1000.0) / 1000.0
                s.set_pose(x=0.0, y=0.0, z=0.02, yaw=0.0, roll=0.35)
                for _ in range(2):                                    # put the lowest gel edge over the target
                    C = _box_corners(d.geom_xpos[gid], d.geom_xmat[gid].reshape(3, 3), m.geom_size[gid])
                    low = C[np.argsort(C[:, 2])[:2]]
                    mid = low.mean(0)
                    q = d.qpos
                    s.set_pose(x=q[s.jnt["slide_x"]] + target[0] - mid[0], y=q[s.jnt["slide_y"]] + target[1] - mid[1],
                               z=0.02, yaw=0.0, roll=0.35)
                geoms = ([m.geom(f"{NAME}_visual").id] if label == "convex_hull_of_mesh" else obj_parts(s))

                def touching(z, s=s, geoms=geoms):
                    s.data.qpos[s.jnt["slide_z"]] = z / 1000.0
                    mujoco.mj_kinematics(s.model, s.data)
                    return any(mujoco.mj_geomDistance(s.model, s.data, s.gel_id, g, 0.002, None) < 0 for g in geoms)

                zc = first_contact(12.0, -1.0, touching)
                C = _box_corners(d.geom_xpos[gid], d.geom_xmat[gid].reshape(3, 3), m.geom_size[gid])
                row[label] = None if zc is None else zc
                if label == "convex_hull_of_mesh":
                    # exact: the same pose, gel box vs dense samples of the visual mesh
                    def touching_exact(z, s=s):
                        s.data.qpos[s.jnt["slide_z"]] = z / 1000.0
                        mujoco.mj_kinematics(s.model, s.data)
                        return _box_sdf(pts_w, s.data.geom_xpos[gid], s.data.geom_xmat[gid].reshape(3, 3),
                                        s.model.geom_size[gid]).min() < 0
                    row["exact"] = first_contact(12.0, -1.0, touching_exact)
                    s.data.qpos[s.jnt["slide_z"]] = row["exact"] / 1000.0
                    mujoco.mj_kinematics(s.model, s.data)
                    C = _box_corners(d.geom_xpos[gid], d.geom_xmat[gid].reshape(3, 3), m.geom_size[gid])
                    row["edge_height_at_exact_contact_mm"] = float(np.sort(C[:, 2])[0] * 1000)
            rep["gel_edge"].append(row)
    tine_top = {}
    for row in rep["gel_edge"]:
        e = row["exact"]
        msg = "  ".join(f"{k} {row[k] - e:+.4f}" for k in list(variants) + ["convex_hull_of_mesh"])
        log(f"  gel edge (roll 20 deg) over the {row['gap']:6s} gap, offset {row['offset_mm']:+.2f} mm: exact first "
            f"contact at slide_z {e:.3f} mm (edge {row['edge_height_at_exact_contact_mm']:.3f} mm above the table); "
            f"error [mm]: {msg}")
        check(abs(row["coacd"] - e) < 0.02, f"CoACD: gel edge first contact off by {row['coacd'] - e:.3f} mm")
    REPORT["between_tines"] = rep


def _box_corners(c, R, h):
    s = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
    return (s * h) @ R.T + c


def _box_sdf(P, c, R, h):
    q = np.abs((P - c) @ R) - h
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)


# ---------------------------------------------------------------- c. footprint
def random_press(sim, rng, force, max_tries=20):
    for _ in range(max_tries):
        p = np.full(len(SITES), 0.65 / max(len(SITES) - 1, 1))
        p[0] = 0.35 if len(SITES) > 1 else 1.0
        site = str(rng.choice(SITES, p=p / p.sum()))
        rel = dict(x=float(rng.uniform(-0.005, 0.005)), y=float(rng.uniform(-0.004, 0.004)),
                   yaw=float(rng.uniform(-np.pi, np.pi)), pitch=float(rng.uniform(-0.06, 0.06)),
                   roll=float(rng.uniform(-0.06, 0.06)))
        info = sim.press_relative(force, site=f"{NAME}_{site}", **rel)
        if info.settled and not info.table_contact and not info.housing_contact:
            break
    return info, site, rel


def footprint_stats(sim, thr=0.01):
    m, d = sim.model, sim.data
    s = sim.mm_per_px
    h_vis = sim.indentation_mm(raw=True)
    h_col = indentation_from_group(sim, COLLISION_GROUP)
    A, B = h_vis > thr, h_col > thr
    iou = (A & B).sum() / max((A | B).sum(), 1)
    ys, xs = np.nonzero(A)
    yc, xc = np.nonzero(B)
    cen = np.hypot(xs.mean() - xc.mean(), ys.mean() - yc.mean()) * s if len(xs) and len(xc) else np.nan
    # mean symmetric distance between the region boundaries
    def edge(M):
        return cv2.morphologyEx(M.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    ea, eb = edge(A), edge(B)
    dta = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 5)
    dtb = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 5)
    bd = 0.5 * (dtb[ea].mean() + dta[eb].mean()) * s if ea.any() and eb.any() else np.nan
    # MuJoCo contact points (gel vs fork parts) in the image
    # (against ANY visual penetration: MuJoCo also reports contacts only a few um deep)
    R, p0 = d.cam_xmat[sim.cam_id].reshape(3, 3), d.cam_xpos[sim.cam_id]
    H, W = h_vis.shape
    dtA = cv2.distanceTransform((~(h_vis > 1e-3)).astype(np.uint8), cv2.DIST_L2, 5)
    hmax = cv2.dilate(h_vis.astype(np.float32), np.ones((5, 5), np.uint8))
    cp, pen, frc, inside, ddiff, side = [], [], [], [], [], 0
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if sim.gel_id not in (c.geom1, c.geom2):
            continue
        q = R.T @ (c.pos - p0)
        col, row = (W - 1) / 2 + q[0] * 1000 / s, (H - 1) / 2 - q[1] * 1000 / s
        mujoco.mj_contactForce(m, d, i, f6)
        cp.append((col, row))
        pen.append(-c.dist * 1000)
        frc.append(f6[0])
        ci, ri = min(max(int(round(col)), 0), W - 1), min(max(int(round(row)), 0), H - 1)   # border: clamp
        inside.append(dtA[ri, ci] * s <= 0.15)
        if abs(np.dot(c.frame[:3], R[:, 2])) > 0.9:     # normal along the camera axis: compare depths
            ddiff.append(abs(-c.dist * 1000 - hmax[ri, ci]))
        else:                                           # gel rim cutting a tine: pushed out sideways
            side += 1
    cp, frc, inside = np.array(cp), np.array(frc), np.array(inside, bool)
    return dict(iou=float(iou), centroid_offset_mm=float(cen), boundary_mean_dist_mm=float(bd),
                area_visual_mm2=float(A.sum() * s * s), area_physics_mm2=float(B.sum() * s * s),
                peak_visual_mm=float(h_vis.max()), peak_physics_geometry_mm=float(h_col.max()),
                n_contacts=int(len(cp)), contacts_on_visual_region=float(np.mean(inside)) if len(cp) else np.nan,
                contact_force_on_visual_region=float(frc[inside].sum() / max(frc.sum(), 1e-12)) if len(cp) else np.nan,
                contact_depth_vs_visual_mm_max=float(max(ddiff)) if ddiff else 0.0, sideways_rim_contacts=side,
                max_contact_penetration_mm=float(max(pen)) if pen else 0.0), h_vis, h_col, cp


def check_footprint(sim, out_dir, n=20):
    log("\n== c. footprint: physics geometry vs depth render of the visual mesh ==")
    sim.place_object(NAME)
    rng = np.random.default_rng(11)
    rows, tiles = [], []
    for k in range(n):
        F = float(np.exp(rng.uniform(np.log(0.5), np.log(5.0))))
        info, site, rel = random_press(sim, rng, F)
        st, h_vis, h_col, cp = footprint_stats(sim)
        st.update(force=F, gel_force=info.gel_force, site=site, settled=info.settled, table=info.table_contact)
        rows.append(st)
        log(f"  {k:2d} {site:10s} F {F:4.2f} N (gel {info.gel_force:4.2f}): IoU {st['iou']:.4f}, centroid offset "
            f"{st['centroid_offset_mm'] * 1000:5.1f} um, boundary {st['boundary_mean_dist_mm'] * 1000:5.1f} um, peak "
            f"{st['peak_visual_mm']:.3f}/{st['peak_physics_geometry_mm']:.3f} mm (visual/physics), "
            f"{st['n_contacts']} contacts, {st['contacts_on_visual_region'] * 100:.0f}% on the visual footprint, "
            f"depth within {st['contact_depth_vs_visual_mm_max'] * 1000:.0f} um of the visual map there")
        A, B = h_vis > 0.01, h_col > 0.01
        img = np.full(A.shape + (3,), 30, np.uint8)
        img[A & B] = (200, 200, 200)
        img[B & ~A] = (40, 40, 255)
        img[A & ~B] = (255, 140, 0)
        for col, row in cp:
            cv2.circle(img, (int(round(col)), int(round(row))), 3, (60, 220, 60), -1, cv2.LINE_AA)
        tiles.append(np.concatenate([_bar(320, f"{site} {F:.1f} N  IoU {st['iou']:.3f}", scale=0.45), img], 0))
    iou = np.array([r["iou"] for r in rows])
    summ = dict(n=n, iou_mean=float(iou.mean()), iou_min=float(iou.min()),
                centroid_offset_mm_mean=float(np.nanmean([r["centroid_offset_mm"] for r in rows])),
                centroid_offset_mm_max=float(np.nanmax([r["centroid_offset_mm"] for r in rows])),
                boundary_mean_dist_mm=float(np.nanmean([r["boundary_mean_dist_mm"] for r in rows])),
                contacts_on_visual_region=float(np.nanmean([r["contacts_on_visual_region"] for r in rows])),
                contact_force_on_visual_region=float(np.nanmean([r["contact_force_on_visual_region"] for r in rows])),
                contact_depth_vs_visual_mm_max=float(max(r["contact_depth_vs_visual_mm_max"] for r in rows)),
                sideways_rim_contacts=int(sum(r["sideways_rim_contacts"] for r in rows)),
                n_contacts=int(sum(r["n_contacts"] for r in rows)),
                peak_diff_mm_max=float(max(abs(r["peak_visual_mm"] - r["peak_physics_geometry_mm"]) for r in rows)),
                contact_penetration_vs_peak_visual_mm_max=float(max(abs(r["max_contact_penetration_mm"] - r["peak_visual_mm"])
                                                                    for r in rows)))
    log(f"  => IoU mean {summ['iou_mean']:.4f} (min {summ['iou_min']:.4f}), centroid offset mean "
        f"{summ['centroid_offset_mm_mean'] * 1000:.1f} um (max {summ['centroid_offset_mm_max'] * 1000:.1f}), boundary "
        f"{summ['boundary_mean_dist_mm'] * 1000:.1f} um; MuJoCo contact points: "
        f"{summ['contacts_on_visual_region'] * 100:.1f}% (force-weighted {summ['contact_force_on_visual_region'] * 100:.2f}%) "
        f"on the visual footprint; penetration of each of the {summ['n_contacts'] - summ['sideways_rim_contacts']} "
        f"contacts facing the gel within {summ['contact_depth_vs_visual_mm_max'] * 1000:.0f} um of the visual depth map there "
        f"({summ['sideways_rim_contacts']} sideways contacts where the gel's rim cuts a tine); deepest contact vs peak of the map within "
        f"{summ['contact_penetration_vs_peak_visual_mm_max'] * 1000:.0f} um")
    check(summ["iou_min"] > 0.95 and summ["centroid_offset_mm_max"] < 0.05, "footprint mismatch")
    check(summ["contacts_on_visual_region"] > 0.99 and summ["contact_force_on_visual_region"] > 0.999,
          "MuJoCo contacts outside the visual footprint")
    grid = np.concatenate([np.concatenate(tiles[i:i + 5], 1) for i in range(0, len(tiles), 5)], 0)
    title = _bar(grid.shape[1], "footprint at the settled pose: grey = both, red = physics geometry only, blue = visual "
                 "only, green dots = MuJoCo contact points (tactile camera view, 0.1 mm/px)")
    cv2.imwrite(os.path.join(out_dir, "footprint.png"), np.concatenate([title, grid], 0))
    REPORT["footprint"] = dict(summary=summ, presses=rows)


# ---------------------------------------------------------------- d. sanity
def hold(sim, n=600):
    """After a press: keep stepping, return jitter (slide_z std/peak-to-peak [um], touch std [N])
    and the stepping speed [steps/s]."""
    m, d = sim.model, sim.data
    z, f = np.empty(n), np.empty(n)
    t = time.perf_counter()
    for k in range(n):
        mujoco.mj_step(m, d)
        z[k], f[k] = d.qpos[sim.jnt["slide_z"]], d.sensordata[sim.touch_adr]
    sps = n / (time.perf_counter() - t)
    return dict(z_std_um=float(z.std() * 1e6), z_p2p_um=float(np.ptp(z) * 1e6), force_std_N=float(f.std()),
                steps_per_s=float(sps))


def force_sweep(sim, sites, forces, label):
    rows = []
    for site in sites:
        for F in forces:
            info = sim.press_relative(F, site=f"{NAME}_{site}")
            j = hold(sim)
            err = abs(info.gel_force - F)
            ok = (err <= max(0.05 * F, 0.03) and 0.2 <= info.max_indent_mm <= 2.0 and info.settled
                  and not info.table_contact and not info.housing_contact and j["z_std_um"] < 2.0
                  and j["force_std_N"] < max(0.02 * F, 0.01) and info.object_shift_mm < 2.0)
            rows.append(dict(site=site, force=F, gel_force=info.gel_force, depth_mm=info.max_indent_mm,
                             settled=info.settled, steps=info.steps, table=info.table_contact,
                             housing=info.housing_contact, tines=info.visible_tines,
                             object_shift_mm=info.object_shift_mm, ncon=int(sim.data.ncon), ok=bool(ok), **j))
    n_ok = sum(r["ok"] for r in rows)
    err = [abs(r["gel_force"] / r["force"] - 1) for r in rows]
    log(f"  {label}: {n_ok}/{len(rows)} presses OK; touch error median {np.median(err) * 100:.2f}% max "
        f"{max(err) * 100:.2f}%; depth {min(r['depth_mm'] for r in rows):.2f}-{max(r['depth_mm'] for r in rows):.2f} mm; "
        f"jitter slide_z std <= {max(r['z_std_um'] for r in rows):.2f} um, touch std <= "
        f"{max(r['force_std_N'] for r in rows):.4f} N")
    for r in rows:
        if not r["ok"]:
            log(f"    not OK: {r['site']} {r['force']} N -> gel {r['gel_force']:.3f} N, depth {r['depth_mm']:.3f} mm, "
                f"settled {r['settled']}, table {r['table']}, housing {r['housing']}, z std {r['z_std_um']:.2f} um, "
                f"F std {r['force_std_N']:.4f}, shift {r['object_shift_mm']:.2f} mm")
    return rows


def speed(sim, tool, F=3.0, site=None, reps=3, n=2000):
    if sim.tool != tool:
        sim.place_object(tool)
    sim.press_relative(F, site=site)
    return max(hold(sim, n)["steps_per_s"] for _ in range(reps))


def check_sanity(out_dir):
    log("\n== d. sanity: force, depth, settling, jitter, speed ==")
    rep = {}
    sim = new_sim(seed=1)
    sim.place_object(NAME)
    forces = [0.5, 2.0, 5.0, 10.0]
    sites = ["face", "palm", "neck", "handle"] if NAME == "fork_stl" else SITES
    rep["static"] = force_sweep(sim, sites, forces, f"static {NAME}")
    check(all(r["ok"] for r in rep["static"]), "static force sweep")
    sim.place_object("fork_dinner")
    rep["baseline_fork_dinner"] = force_sweep_old(sim, forces)

    # freejoint version: presses on the tines must be clean; elsewhere see KNOWN_FREE_ISSUE
    sf = new_sim(seed=1, free_objects=True)
    sf.place_object(NAME)
    rep["free_face"] = force_sweep(sf, ["face"], forces, f"freejoint {NAME}, face site")
    check(all(r["ok"] for r in rep["free_face"]), "freejoint force sweep on the face site")
    rest = [x for x in sites if x != "face"]
    if rest:
        rep["free_other_sites"] = force_sweep(sf, rest, forces, f"freejoint {NAME}, {'/'.join(rest)} "
                                              f"({'known issue, ' if NAME == 'fork_stl' else ''}not judged)")
    if NAME == "fork_stl":
        rep["known_free_issue"] = KNOWN_FREE_ISSUE

    # speed before / after adding the fork (same active object; interleaved, best of 3)
    before = new_sim(seed=1, spec_edit=remove_object)
    cases = [("before: fork_dinner pressed 3 N", before, "fork_dinner", None, True),
             (f"after:  fork_dinner pressed 3 N ({NAME} parked)", sim, "fork_dinner", None, True),
             ("before: fork_dinner active, sensor in the air", before, "fork_dinner", None, False),
             ("after:  fork_dinner active, sensor in the air", sim, "fork_dinner", None, False),
             (f"after:  {NAME} active, sensor in the air", sim, NAME, None, False)]
    cases += [(f"after:  {NAME} {x} pressed 3 N", sim, NAME, f"{NAME}_{x}", True) for x in sites[:3]]
    cases += [(f"after:  freejoint {NAME} face pressed 3 N", sf, NAME, f"{NAME}_face", True)]
    sp = {c[0]: 0.0 for c in cases}
    for _ in range(3):
        for label, s, tool, site, pressed in cases:
            if s.tool != tool:
                s.place_object(tool)
            if pressed:
                s.press_relative(3.0, site=site)
                v = hold(s, 2000)["steps_per_s"]
            else:
                v = _idle_speed(s)
            sp[label] = max(sp[label], v)
    rep["steps_per_s"] = sp
    log("  steps/s (best of 3 x 2000-3000 steps, single thread):")
    for k, v in sp.items():
        log(f"    {k:52s} {v:9,.0f}")
    REPORT["sanity"] = rep
    return sim


KNOWN_FREE_ISSUE = (
    "freejoint fork_stl pressed on the palm, neck or handle can rock (touch force swings up to ~0.5-3 N, "
    "slide_z by ~5-30 um; the handle at >= 5 N may not settle). Cause: each tine tip rests on the table "
    "along a level edge and MuJoCo's plane-mesh collision reports one vertex per convex part, so the "
    "support point flips between the edge's two corners. Contact margins or explicit table/part pairs "
    "steady it but shift the gel's force-depth law by 20-250 um, so they are not used. The default "
    "(static) fork and freejoint presses on the tines are not affected.")


def force_sweep_old(sim, forces):
    rows = []
    for F in forces:
        info = sim.press_relative(F)
        j = hold(sim)
        rows.append(dict(force=F, gel_force=info.gel_force, depth_mm=info.max_indent_mm, **j))
    log(f"  baseline fork_dinner (tines): depth {', '.join(f'{r['depth_mm']:.2f}' for r in rows)} mm at "
        f"{forces} N; jitter slide_z std <= {max(r['z_std_um'] for r in rows):.2f} um, touch std <= "
        f"{max(r['force_std_N'] for r in rows):.4f} N")
    return rows


def _idle_speed(sim, n=3000):
    sim.lift()
    t = time.perf_counter()
    for _ in range(n):
        mujoco.mj_step(sim.model, sim.data)
    return n / (time.perf_counter() - t)


# ---------------------------------------------------------------- e. gallery
GALLERY = [  # (site, title, press_relative kwargs)
    ("face", "tines, 3 N", dict(force=3.0)),
    ("face", "tines turned 90 deg, 3 N", dict(force=3.0, yaw=np.pi / 2)),
    ("face", "tines, light 0.5 N", dict(force=0.5, yaw=0.3)),
    ("face", "tines, tilted 4.6 deg to the palm, 4 N", dict(force=4.0, pitch=-0.08, yaw=0.4)),
    ("tips", "tine tips, pad across, 2 N", dict(force=2.0, yaw=np.pi / 2, y=0.006)),
    ("face", "2 tines on the pad, 2 N", dict(force=2.0, y=0.013)),
    ("palm", "palm + tine roots, 2 N", dict(force=2.0, x=0.002)),
    ("palm", "palm, 6 N, turned", dict(force=6.0, yaw=0.7)),
    ("neck", "neck, 3 N", dict(force=3.0, x=-0.004, yaw=-0.3)),
    ("handle", "handle, 3 N", dict(force=3.0, yaw=0.3)),
    ("handle_end", "handle end, 4 N", dict(force=4.0, x=0.005, yaw=2.0)),
    ("handle", "handle, 10 N, rolled", dict(force=10.0, roll=0.05, yaw=-1.0)),
]


def check_gallery(sim, out_dir, det):
    log("\n== e. gallery ==")
    from fork_detector import overlay
    sim.place_object(NAME)
    ref = sim.reference_image()
    vis = visual_mesh()
    R, p = body_pose(sim)
    tiles, rows = [], []
    gallery = GALLERY if is_fork_stl() else [(x, f"{x}, {F:g} N", dict(force=F)) for x in SITES for F in (1.0, 4.0)][:12]
    for site, title, kw in gallery:
        kw = dict(kw)
        F = kw.pop("force")
        info = sim.press_relative(F, site=f"{NAME}_{site}", **kw)
        img = sim.tactile_image()
        h = sim.indentation_mm(raw=True)
        res = det.detect(img, ref) if det is not None else None
        truth_fork = NAME in FORKS and info.visible_tines >= 2
        ok = res is None or (res.is_fork == truth_fork if res.contact else False)
        head = _bar(320, f"{title}", scale=0.48)
        head2 = _bar(320, f"gel {info.gel_force:.2f} N  peak {info.max_indent_mm:.2f} mm  {info.visible_tines} tines",
                     scale=0.42, color=(200, 200, 200), h=20)
        tac = overlay(img, res, sim.mm_per_px, scale=1) if res is not None else img[..., ::-1]
        dep = cv2.applyColorMap((np.clip(h / 1.5, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        verdict = ("" if res is None else ("no contact" if not res.contact else
                                           f"detector: {'FORK' if res.is_fork else 'not fork'} p={res.p_fork:.2f}"))
        foot = _bar(320, f"{verdict} ({'ok' if ok else 'WRONG'}: {'fork' if truth_fork else '<2 tines'})", scale=0.42,
                    color=(80, 220, 80) if ok else (60, 60, 255), h=20)
        loc = _locator(sim, vis, R, p, world_bbox_mm(sim))
        tile = np.concatenate([head, head2, tac, dep, loc, foot], 0)
        tiles.append(cv2.copyMakeBorder(tile, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(40, 40, 40)))
        rows.append(dict(site=site, title=title, force=F, gel_force=info.gel_force, depth_mm=info.max_indent_mm,
                         visible_tines=info.visible_tines, settled=info.settled, table=info.table_contact,
                         housing=info.housing_contact,
                         detector=None if res is None else dict(contact=bool(res.contact), is_fork=bool(res.is_fork),
                                                                p_fork=float(res.p_fork), n_tines=int(res.n_tines))))
        log(f"  {title:34s} gel {info.gel_force:5.2f} N  peak {info.max_indent_mm:.2f} mm  tines {info.visible_tines}  "
            f"settled {info.settled}  table {info.table_contact}  housing {info.housing_contact}  {verdict}")
        check(info.settled and not info.table_contact and not info.housing_contact
              and 0.2 <= info.max_indent_mm <= 2.0, f"gallery press '{title}'")
    grid = np.concatenate([np.concatenate(tiles[i:i + 6], 1) for i in range(0, len(tiles), 6)], 0)
    title = _bar(grid.shape[1], f"{NAME} ({REPORT['mesh']['source']}) on the table, sensor pressing from above | per tile: "
                 "tactile RGB + detector overlay, ground-truth gel indentation (0-1.5 mm), where the pad is", h=30)
    cv2.imwrite(os.path.join(out_dir, "gallery.png"), np.concatenate([title, grid], 0))
    REPORT["gallery"] = rows


def _locator(sim, vis, R, p, bbox, W=320, H=74):
    """Top view of the object (world-aligned) with the tactile camera's field of view."""
    img = np.full((H, W, 3), 30, np.uint8)
    v = (vis.vertices @ R.T + p * 1000.0)[:, :2]
    (lx, ly, _), (hx, hy, _) = bbox
    x0, x1, y0 = lx - 4.0, hx + 4.0, 0.5 * (ly + hy)
    s = W / (x1 - x0)
    to_px = lambda P: np.c_[(P[:, 0] - x0) * s, H / 2 - (P[:, 1] - y0) * s].round().astype(np.int32)
    P = to_px(v)
    for f in vis.faces:
        cv2.fillPoly(img, [P[f]], (160, 160, 160))
    H_img, W_img = sim.opt.height, sim.opt.width
    cs, rs = [0, W_img - 1, W_img - 1, 0], [0, 0, H_img - 1, H_img - 1]
    corners = np.array([sim.px_to_world_mm(c, r) for c, r in zip(cs, rs)])
    cv2.polylines(img, [to_px(corners)], True, (0, 200, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------- f. SDF
def check_sdf(out_dir, coacd_sim):
    log("\n== f. MuJoCo native mesh SDF instead of CoACD ==")
    rep = {}
    for depth in (None, 7):
        label = f"sdf_octree_depth{depth or 6}"
        t = time.perf_counter()
        s = new_sim(seed=1, spec_edit=sdf_variant(depth))
        compile_s = time.perf_counter() - t
        s.place_object(NAME)
        r = dict(compile_s=compile_s, octree_nodes=int(s.model.noct))
        # geometry: height of the gel-facing surface along vertical rays, vs the visual mesh
        r["rays"] = _ray_compare(s, coacd_sim)
        r["presses"] = force_sweep(s, (["face", "palm", "handle"] if NAME == "fork_stl" else SITES[:3]),
                                   [0.5, 2.0, 5.0], label)
        r["steps_per_s_tines_3N"] = speed(s, NAME, site=f"{NAME}_face")
        log(f"  {label}: compile {compile_s:.2f} s, {s.model.noct} octree nodes, {r['steps_per_s_tines_3N']:,.0f} "
            f"steps/s at 3 N on the tines")
        rep[label] = r
    rep["coacd_rays"] = _ray_compare(coacd_sim, coacd_sim)
    log(f"  CoACD, same ray test: gel-side error max {rep['coacd_rays']['max_mm']:.3f} mm, p99 "
        f"{rep['coacd_rays']['p99_mm']:.4f}, gap area filled {rep['coacd_rays']['filled_mm2']:.2f} mm2")
    REPORT["sdf"] = rep


def _ray_compare(sim, ref_sim, step=0.1):
    """mj_ray straight down onto the collision geometry (group 3) over the fork head, vs the
    visual mesh (group 1): what the gel would touch."""
    out = {}
    m, d = sim.model, sim.data
    lo, hi = world_bbox_mm(sim)
    if NAME == "fork_stl":                    # the head: tines, palm, neck
        lo, hi = np.array([-6.0, lo[1], 0]), np.array([38.0, hi[1], 0])
    xs = np.arange(lo[0] - 1.0, hi[0] + 1.0, step)
    ys = np.arange(lo[1] - 1.0, hi[1] + 1.0, step)
    gid = np.zeros(1, np.int32)

    def hmap(s, group):
        grp = np.zeros(6, np.uint8)
        grp[group] = 1
        H = np.full((len(ys), len(xs)), np.nan)
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                dist = mujoco.mj_ray(s.model, s.data, np.array([x / 1000, y / 1000, 0.02]), np.array([0, 0, -1.0]),
                                     grp, 1, -1, gid, None)
                if dist >= 0:
                    H[i, j] = 20.0 - dist * 1000
        return H
    t = time.perf_counter()
    with show_parts(sim):                     # only this object's parts in group 3
        hc = hmap(sim, COLLISION_GROUP)
    out["ray_time_s"] = time.perf_counter() - t
    hv = hmap(ref_sim, ACTIVE_GROUP)
    both = ~np.isnan(hv) & ~np.isnan(hc)
    dev = hc - hv
    rim = cv2.dilate((~np.isnan(hv)).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    filled = np.isnan(hv) & ~np.isnan(hc)
    out.update(max_mm=float(np.abs(dev[both]).max()), p99_mm=float(np.percentile(np.abs(dev[both]), 99)),
               mean_abs_mm=float(np.abs(dev[both]).mean()), filled_mm2=float(filled.sum() * step ** 2),
               filled_beyond_rim_mm2=float((filled & ~rim).sum() * step ** 2),
               missing_mm2=float((~np.isnan(hv) & np.isnan(hc)).sum() * step ** 2))
    if sim is not ref_sim:
        log(f"  {'':2s}rays onto the SDF: gel-side error max {out['max_mm']:.3f} mm, p99 {out['p99_mm']:.4f}, "
            f"mean {out['mean_abs_mm']:.4f}; gap area filled {out['filled_mm2']:.2f} mm2, missing {out['missing_mm2']:.2f} mm2")
    return out


# ---------------------------------------------------------------- g. detector
def check_detector(n, workers):
    log("\n== g. existing fork detectors on fork_stl presses ==")
    import train_detector as td
    from fork_detector import ForkDetector
    rep = {}
    for tag in ("results_moving", "results"):
        path = os.path.join(HERE, tag, "fork_detector.joblib")
        if not os.path.exists(path):
            continue
        det = ForkDetector.load(path)
        cfg = dict(rel_thr=det.rel_thr, count_feature=det.count_feature, contact_depth_mm=det.contact_depth_mm,
                   contact_area_mm2=det.contact_area_mm2, evidence_level=det.evidence_level,
                   evidence_mm2=det.evidence_mm2)
        g = td.generate([NAME], np.array([1.0]), n, 5, det.pm.A, det.pm.b, workers, cfg=cfg, mode="moving_sensor")
        rows = g["rows"]
        gt = np.array([r["gt_contact"] for r in rows])
        dc = np.array([r["det_contact"] for r in rows])
        hov = np.array([r["hover"] for r in rows])
        tines = np.array([r["visible_tines"] for r in rows])
        p = np.zeros(len(rows))
        sel = dc
        p[sel] = det.clf.predict_proba(g["X"][sel])[:, 1]
        pred = p >= det.threshold
        ev = gt & dc & (tines >= 2)
        rep[tag] = dict(n=n, contact_accuracy=float((gt == dc).mean()), hover=int(hov.sum()),
                        hover_no_contact=int((hov & ~dc).sum()), recall_tines_ge2=float(pred[ev].mean()),
                        n_tines_ge2=int(ev.sum()),
                        by_tines={str(k): dict(n=int((gt & dc & (tines == k)).sum()),
                                               called_fork=float(pred[gt & dc & (tines == k)].mean())
                                               if (gt & dc & (tines == k)).any() else None) for k in range(5)})
        log(f"  {tag}/fork_detector.joblib: recall {rep[tag]['recall_tines_ge2'] * 100:.1f}% on {ev.sum()} presses with "
            f">= 2 tines on the pad; contact detection {rep[tag]['contact_accuracy'] * 100:.1f}%; hover "
            f"{rep[tag]['hover_no_contact']}/{rep[tag]['hover']} read 'no contact'; called fork by tines on pad: "
            + ", ".join(f"{k}: {v['called_fork']:.2f} (n={v['n']})" for k, v in rep[tag]["by_tines"].items()
                        if v["n"]))
    REPORT["detector"] = rep


# ---------------------------------------------------------------- h. regression
def check_regression():
    log(f"\n== h. regression: original objects with and without {NAME} in the model ==")
    bad, n = [], 0
    for free in (False, True):
        a = new_sim(seed=5, free_objects=free, spec_edit=remove_object)
        b = new_sim(seed=5, free_objects=free)
        for k, t in enumerate(TOOLS):
            for s in (a, b):
                s.place_object(t, x=0.01 * (k % 3) - 0.01, y=0.004 * (k % 4) - 0.006, yaw=0.4 * k - 2.0)
            for F, rel in ((0.7, dict(x=0.002, yaw=0.3)), (4.0, dict(y=-0.002, yaw=-0.8, pitch=0.04, roll=-0.03))):
                ia, ib = a.press_relative(F, **rel), b.press_relative(F, **rel)
                same = (ia.gel_force == ib.gel_force and ia.steps == ib.steps
                        and np.array_equal(a.data.qpos[:6], b.data.qpos[:6])
                        and np.array_equal(a.tactile_image(), b.tactile_image()))
                n += 1
                if not same:
                    bad.append(f"{'free' if free else 'static'} {t} {F} N")
    log(f"  {n - len(bad)}/{n} presses bit-identical (gel force, steps, arm pose, tactile image)")
    check(not bad, f"regression: {bad}")
    REPORT["regression"] = dict(n=n, differ=bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="fork_stl", help="object made by stl_to_mjcf.py (meshes/<name>/)")
    ap.add_argument("--xml", default=None, help="model with the object (default gelsight_moving.xml)")
    ap.add_argument("--out", default=None, help="default: figures_stl/ (fork_stl) or figures_stl/<name>/")
    ap.add_argument("--detector-n", type=int, default=400, help="fork_stl presses per detector (0 = skip)")
    ap.add_argument("--workers", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    ap.add_argument("--skip", default="", help="comma-separated sections to skip (rest,a,b,c,d,e,f,g,h)")
    args = ap.parse_args()
    global NAME, MESH_DIR, SITES, XML
    NAME, XML = args.name, args.xml
    MESH_DIR = os.path.join(HERE, "meshes", NAME)
    args.out = args.out or os.path.join(HERE, "figures_stl", *([] if NAME == "fork_stl" else [NAME]))
    os.makedirs(args.out, exist_ok=True)
    skip = set(args.skip.split(",")) if args.skip else set()
    t0 = time.time()
    REPORT["mesh"] = json.load(open(os.path.join(MESH_DIR, "report.json")))
    views = new_sim(seed=0, spec_edit=edits(add_views, add_probe))
    views.place_object(NAME)
    SITES = object_sites(views)
    log(f"{NAME}: sites {SITES}; {'fork checks on' if is_fork_stl() else 'no tines: fork-specific checks off'}")
    if "rest" not in skip:
        check_rest(views)
    if "a" not in skip:
        check_overlay(views, args.out)
    if "b" not in skip and is_fork_stl():
        variants = {"coacd": views}
        if "f" not in skip:
            variants["sdf"] = new_sim(seed=0, spec_edit=edits(add_probe, sdf_variant()))
        check_gaps(args.out, variants)
    if "c" not in skip:
        check_footprint(new_sim(seed=2), args.out)
    if "d" not in skip:
        check_sanity(args.out)
    if "e" not in skip:
        from fork_detector import ForkDetector
        path = os.path.join(HERE, "results_moving", "fork_detector.joblib")
        det = ForkDetector.load(path) if os.path.exists(path) else None
        check_gallery(new_sim(seed=3), args.out, det)
    if "f" not in skip:
        check_sdf(args.out, views)
    if "g" not in skip and args.detector_n > 0 and XML is None:
        check_detector(args.detector_n, args.workers)
    if "h" not in skip:
        check_regression()
    REPORT["failed_checks"] = FAIL
    REPORT["elapsed_s"] = time.time() - t0
    with open(os.path.join(args.out, "verify_stl_report.json"), "w") as f:
        json.dump(REPORT, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    log(f"\n{len(FAIL)} failed checks ({time.time() - t0:.0f} s); report: "
        f"{os.path.relpath(os.path.join(args.out, 'verify_stl_report.json'), HERE)}")
    for f in FAIL:
        log("  -", f)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
