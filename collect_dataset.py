"""
Tactile dataset of the STL fork (fork_stl = ForkMilimeter.stl, CoACD collision parts) for a
model that (1) recognises the fork and (2) predicts where the fork's centre lies relative to the
sensor ("which way to move"). Moving-sensor simulator (gelsight_moving.xml).

    python collect_dataset.py poses               # 1. stable resting poses -> dataset/poses/
    python collect_dataset.py collect             # 2+3. 10,000 samples, 8k/1k/1k -> dataset/<split>-NNNN.h5
    python collect_dataset.py verify              # 4. labels, statistics, physics, reload -> dataset/verify/
    python collect_dataset.py regen dataset/val-0000.h5 7     # regenerate sample 7 from its seed, compare

The fork is clamped to the table (it never moves during a press; between samples it lies at another
place, yaw and resting pose). By default only the natural resting poses are sampled
(--include-edge-poses adds the propped side-edge poses) and no free-fork run is made (--free-run
adds one, for would_move). The shards in dataset/ were made before these defaults: they contain 10 %
propped samples and the free-run fields.

Every sample is reproducible from its seed (plus the shard's reference image, whose seed is
stored too). Conventions, units and the HDF5 layout are written into every shard
("metadata_json") and into dataset/metadata.json; README.md describes them.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"          # offscreen rendering; must precede `import mujoco`

import cv2
import numpy as np

import objects as _objects

HERE = os.path.dirname(os.path.abspath(__file__))
# The object comes from the registry (objects.json, objects.py); fork_stl is the default. use_object() switches to
# another registered object (the multi-object collector, collect_multi.py), like set_sensor() switches the sensor.
NAME = "fork_stl"
MESH_DIR = _objects.mesh_dir(NAME)
SCENE_XML = None       # None: GelSightSim's default scene (gelsight_moving.xml); use_object(..., scene=...) sets another
OUT = os.path.join(HERE, "dataset")
DETECTOR = os.path.join(HERE, "results_moving", "fork_detector.joblib")   # its ball calibration -> depth_recon

# the object's parts, by position along the STL x axis (mm) and by surface normal (see README): from the registry.
# fork_stl: tines >= 7.0 > palm >= 1.0 > neck >= -15.0 > handle, |normal . z| < 0.5 = edge (the constants this file
# had before objects.json; TINE_ROOT_MM etc. are kept for readers of the old names)
PARTS = _objects.part_names(NAME) + ["edge"]
X_STARTS_MM = _objects.part_starts_mm(NAME)
TINE_ROOT_MM, PALM_START_MM, NECK_START_MM = X_STARTS_MM
EDGE_NZ = _objects.edge_nz(NAME)   # |normal . STL z| below this: side wall / rim of the 1.5 mm thick fork -> "edge"
SAMPLE_TYPES = ["on_fork", "table", "hover"]
CATEGORIES = ["natural", "propped", "rejected"]
SPLITS = ["train", "val", "test", "train2", "test2"]   # train2 / test2: the 40k + 5k presses of dataset_v3/
SEED_BASE = 10_000_000         # sample seed = split_index * SEED_BASE + sample index
REF_SEED_BASE = 1_000_000_000  # reference image of shard k of split s: REF_SEED_BASE + 1000 * s + k
MOVE_MM, MOVE_DEG = 0.5, 1.0   # "moved": stability test and would_move
RETRY = {"housing": 1, "unsettled": 2, "fork_touch": 4, "no_free_spot": 8}

CONVENTIONS = {
    "units": "SI (m, N, s, rad) unless a name ends in _mm or _deg; depth maps in mm; images uint8 RGB",
    "quaternion": "(w, x, y, z), unit, rotates vectors from the child frame into the parent frame",
    "rpy": "(roll, pitch, yaw) in rad with R = Rz(yaw) @ Ry(pitch) @ Rx(roll) (intrinsic Z-Y'-X'')",
    "pose_X_in_Y": "p_Y = R @ p_X + t, t = origin of X in Y, R = axes of X in Y (columns)",
    "world_frame": "W: table top is z = 0, +z up",
    "fork_frame": "F: body fork_stl = the STL file's own frame scaled to metres; +x towards the tine "
                  "tips, +z = side the tines curve towards, origin = STL origin (neck/palm junction)",
    "sensor_frame": "S: origin at the centre of the undeformed gel contact face; +x = image column, "
                    "+y = image row, +z = viewing direction, out of the gel into the object (world down "
                    "for an untilted sensor). Right-handed.",
    "pixel_to_sensor": "x_mm = (col - (W-1)/2) * mm_per_px, y_mm = (row - (H-1)/2) * mm_per_px; pixel "
                       "(col, row) shows the point (x, y) of the gel face, the camera is orthographic along +z_S",
    "fork_centre": "bbox: centre of the fork mesh's axis-aligned bounding box in F; com: centre of mass",
    "gel_contacts": "normal_s: unit vector from the fork/table into the gel (= direction of the normal "
                    "force on the gel), in S; force_s: total force the fork/table exerts on the gel, in S; "
                    "fn = normal, ft = tangential (2 components in MuJoCo's contact frame)",
    "table_contacts": "fork-table contacts of the FREE-fork run at the end of the press (the welded fork "
                      "has none): normal_w from the table into the fork, in W",
    "marker_field": "marker_rest (N, 2) and marker_shift (N, 2) in pixels (col, row); drawn at rest + shift",
    "parts": f"by fork-frame x: tines >= {TINE_ROOT_MM} mm > palm >= {PALM_START_MM} > neck >= "
             f"{NECK_START_MM} > handle; 'edge' = surface whose normal is within "
             f"{np.degrees(np.arccos(EDGE_NZ)):.0f} deg of the fork plane (side walls, rim)",
}


@dataclass
class Config:
    frac_on_fork: float = 0.7
    frac_table: float = 0.2
    frac_hover: float = 0.1
    propped_frac: float = 0.1            # share of samples in "propped" poses (held by a fixture); the
                                         # collect CLI sets 0 unless --include-edge-poses
    free_run: bool = True                # repeat each press with a FREE fork -> would_move. The CLI default is
                                         # False: the fork is clamped to the table (it never moves during a press)
    part_weights: dict = field(default_factory=lambda: {p: 1.0 for p in PARTS})
    fork_xy_range_m: float = 0.03        # fork bbox centre uniform in +-range (x and y)
    tilt_max_deg: float = 3.0            # sensor tilt, uniform over the disc of tilts
    force_min_n: float = 0.2
    force_max_n: float = 10.0
    hover_gap_mm: tuple = (0.2, 3.0)
    table_radius_m: float = 0.10         # table-only: sensor centre within this of the fork centre
    table_margin_mm: float = 2.0         # ... and its gel at least this far from the fork's outline
    max_tries: int = 20                  # re-draws of the sensor pose before a flawed press is kept
    free_hold_s: float = 0.3             # hover samples: how long the free fork is watched
    press_max_s: float = 0.6
    sensor: str = "standard"             # gelsight_sim.SENSORS: "standard" 32 x 24 mm, "mini" 18.6 x 14.4 mm


# ---------------------------------------------------------------- small geometry helpers
def quat2mat(q):
    import mujoco
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def mat2quat(R):
    import mujoco
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def mat2rpy(R):
    return np.array([np.arctan2(R[2, 1], R[2, 2]), np.arcsin(np.clip(-R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0])])


def rot_deg(Ra, Rb) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def arm_axes(yaw, pitch, roll):
    """World axes (columns) of the sensor frame S for arm joints yaw/pitch/roll (gelsight_sim)."""
    cp, sp, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    M = rz(yaw) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return M @ np.diag([1.0, -1.0, -1.0])


def part_of(x_mm, nz):
    """Part index for object-frame x (mm) and |normal . object z| (arrays); len(PARTS) - 1 = "edge"."""
    x, nz = np.asarray(x_mm, float), np.abs(np.asarray(nz, float))
    p = _objects.part_from_x(X_STARTS_MM, x)
    return np.where(nz < EDGE_NZ, len(PARTS) - 1, p).astype(np.int8)


def use_object(name, scene=None):
    """Switch this module to another registered object (objects.json): its name, mesh folder, parts and scene.
    scene: the MJCF file to simulate in (default: the registry's scene of the object; None for gelsight_moving.xml's
    fork_stl keeps GelSightSim's default). Spawned workers must call it again (module state)."""
    global NAME, MESH_DIR, SCENE_XML, PARTS, X_STARTS_MM, EDGE_NZ
    NAME = name
    MESH_DIR = _objects.mesh_dir(name)
    sx = scene or _objects.get(name)["scene"]
    SCENE_XML = None if os.path.basename(sx) == "gelsight_moving.xml" else os.path.join(HERE, sx)
    PARTS = _objects.part_names(name) + ["edge"]
    X_STARTS_MM = _objects.part_starts_mm(name)
    EDGE_NZ = _objects.edge_nz(name)


def rasterize(P, F, H, W, keep="min"):
    """Z-buffer of a triangle mesh. P: (n, 3) vertices as (col, row, depth) with pixel centres at
    integer (col, row). Returns depth (H, W) (+-inf where empty) and triangle index (H, W) (-1)."""
    s = 1.0 if keep == "min" else -1.0
    zb = np.full((H, W), np.inf)
    tid = np.full((H, W), -1, np.int32)
    for k, (i, j, l) in enumerate(F):
        a, b, c = P[i], P[j], P[l]
        x0, x1 = max(int(np.ceil(min(a[0], b[0], c[0]))), 0), min(int(np.floor(max(a[0], b[0], c[0]))), W - 1)
        y0, y1 = max(int(np.ceil(min(a[1], b[1], c[1]))), 0), min(int(np.floor(max(a[1], b[1], c[1]))), H - 1)
        if x0 > x1 or y0 > y1:
            continue
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-12:
            continue
        yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        w0 = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / den
        w1 = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        z = s * (w0 * a[2] + w1 * b[2] + w2 * c[2])
        sub = zb[y0:y1 + 1, x0:x1 + 1]
        upd = inside & (z < sub)
        sub[upd] = z[upd]
        tid[y0:y1 + 1, x0:x1 + 1][upd] = k
    return s * zb, tid


def rect_hits_polygon(c, axes, half, poly):
    """Separating-axis test: rectangle (centre c, unit axes (2, 2) as rows, half sizes) vs convex
    polygon (n, 2), all in the same 2-D frame."""
    corners = np.array([c + sx * half[0] * axes[0] + sy * half[1] * axes[1] for sx in (-1, 1) for sy in (-1, 1)])
    e = np.roll(poly, -1, 0) - poly
    normals = np.c_[-e[:, 1], e[:, 0]]
    for n in list(axes) + list(normals):
        n = n / (np.linalg.norm(n) + 1e-15)
        a, b = corners @ n, poly @ n
        if a.max() < b.min() or b.max() < a.min():
            return False
    return True


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------- the fork's geometry
class ForkGeometry:
    def __init__(self):
        import trimesh
        self.vis = trimesh.load(os.path.join(MESH_DIR, f"{NAME}_visual.stl"), process=True)      # mm
        files = sorted(glob.glob(os.path.join(MESH_DIR, f"{NAME}_c*.stl")))
        self.part_names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        self.parts = [trimesh.load(f, process=True) for f in files]
        self.hull_v = np.concatenate([p.vertices for p in self.parts]) / 1000.0                  # m, F
        self.V = self.vis.vertices / 1000.0                                                      # m, F
        self.F = self.vis.faces
        self.bbox_centre = self.vis.bounds.mean(0) / 1000.0                                      # m, F
        self.com = self.vis.center_mass / 1000.0

    def rest_z(self, R):
        """Body height that puts the lowest point of the collision parts (and the mesh) on z = 0."""
        return float(-min((self.hull_v @ R.T)[:, 2].min(), (self.V @ R.T)[:, 2].min()))

    def top_points(self, R, step_mm=0.25, min_nz=0.2):
        """Upward-facing surface seen from above in orientation R: points (F frame, mm) on a
        step_mm grid, with the part of the surface there."""
        P = (self.V @ R.T) * 1000.0
        lo = P[:, :2].min(0) - 1
        W, H = (np.ceil((P[:, :2].max(0) + 1 - lo) / step_mm).astype(int) + 1)
        pix = np.c_[(P[:, 0] - lo[0]) / step_mm, (P[:, 1] - lo[1]) / step_mm, P[:, 2]]
        ztop, tid = rasterize(pix, self.F, H, W, keep="max")
        nz_world = (self.vis.face_normals @ R.T)[:, 2]
        ok = (tid >= 0)
        ok[ok] &= nz_world[tid[ok]] > min_nz
        rows, cols = np.nonzero(ok)
        Pw = np.c_[cols * step_mm + lo[0], rows * step_mm + lo[1], ztop[rows, cols]]
        Pf = Pw @ R                                             # back into the fork frame (R orthonormal)
        # part from the point's own x (a long triangle can span handle, neck and palm) and the
        # surface normal there
        part = part_of(Pf[:, 0], self.vis.face_normals[tid[rows, cols], 2])
        return Pf.astype(np.float32), part.astype(np.int8)


def git_commit():
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


def source_hashes():
    files = ["collect_dataset.py", "gelsight_sim.py", "fork_detector.py", "stl_to_mjcf.py", "gelsight_moving.xml",
             f"meshes/{NAME}/{NAME}.xml", _objects.get(NAME)["stl"]]
    if NAME != "fork_stl":
        files += ["objects.json", "objects.py"] + ([os.path.relpath(SCENE_XML, HERE)] if SCENE_XML else [])
    return {f: sha256_file(os.path.join(HERE, f)) for f in files if os.path.exists(os.path.join(HERE, f))}


def mesh_hash():
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(MESH_DIR, "*.stl"))):
        h.update(os.path.basename(f).encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def model_hash(model):
    import mujoco
    b = np.zeros(mujoco.mj_sizeModel(model), np.uint8)
    mujoco.mj_saveModel(model, None, b)
    return hashlib.sha256(b.tobytes()).hexdigest()


# ================================================================ PART 1: stable poses
def _free_sim():
    from gelsight_sim import GelSightSim
    sim = GelSightSim(xml_path=SCENE_XML, mode="moving_sensor", free_objects=True, seed=0)
    sim.place_object(NAME)
    return sim


def _release(sim, pos, quat, T, trace_every=0):
    """Free fork released at pos/quat with zero velocity, sensor parked far away; T seconds."""
    import mujoco
    m, d = sim.model, sim.data
    a, v = sim._obj_qadr[NAME], sim._obj_dadr[NAME]
    mujoco.mj_resetDataKeyframe(m, d, 0)
    sim.place_object(NAME)
    sim.set_pose(x=0.15, y=0.15, z=0.08)
    d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = pos, quat
    d.qvel[v:v + 6] = 0
    mujoco.mj_forward(m, d)
    tr = []
    for k in range(int(round(T / m.opt.timestep))):
        mujoco.mj_step(m, d)
        if trace_every and k % trace_every == 0:
            tr.append(d.qpos[a:a + 7].copy())
    return d.qpos[a:a + 3].copy(), d.qpos[a + 3:a + 7].copy(), np.array(tr)


def _drop_worker(seeds, obj=None):
    """obj = (name, scene) when the parent switched objects (use_object): a spawned worker starts with fork_stl."""
    from scipy.spatial.transform import Rotation
    if obj is not None:
        use_object(*obj)
    geo = ForkGeometry()
    sim = _free_sim()
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        R = Rotation.random(random_state=s).as_matrix()
        h = rng.uniform(0.002, 0.010)                   # lowest point 2-10 mm above the table
        pos = -R @ geo.bbox_centre
        pos[2] = h - (geo.hull_v @ R.T)[:, 2].min()
        _, q, tr = _release(sim, pos, mat2quat(R), 3.0, trace_every=20)
        Rf = quat2mat(q)
        last = [quat2mat(t[3:]) for t in tr[-20:]]      # last 0.2 s
        out.append(dict(seed=int(s), up_in_fork=(Rf.T @ [0, 0, 1.0]).tolist(),
                        rock_deg=max(rot_deg(last[0], x) for x in last), drop_h_mm=h * 1000))
    return out


def stable_poses(out_dir, n_drops=500, workers=24):
    """Part 1. trimesh candidates -> MuJoCo checks -> natural / propped / rejected."""
    from multiprocessing import get_context
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    from stl_to_mjcf import rest_poses
    os.makedirs(out_dir, exist_ok=True)
    geo = ForkGeometry()
    cands = rest_poses(geo.vis)
    sim = _free_sim()
    t0 = time.time()
    poses = []
    for k, c in enumerate(cands):
        R = c["R"]
        q = mat2quat(R)
        z = geo.rest_z(R)
        bc = R @ geo.bbox_centre
        rest_pos = np.array([-bc[0], -bc[1], z])          # bbox centre over the table origin
        up = R[2]                                       # world +z in the fork frame
        # support polygon of the collision parts and the centre of mass over it
        Vh = geo.hull_v @ R.T + rest_pos
        sup = Vh[Vh[:, 2] < 2e-5]
        com = R @ geo.com + rest_pos
        try:
            hull = ConvexHull(sup[:, :2])
            margin = float(-np.max(hull.equations[:, :2] @ com[:2] + hull.equations[:, 2]))
        except Exception:                                # fewer than 3 support points / collinear
            margin = 0.0
        sup_parts = sorted({PARTS[i] for i in part_of(((sup - rest_pos) @ R)[:, 0] * 1000, np.ones(len(sup)))})
        # (a) the requested test: released 1 mm above the pose, 1 s
        p1, q1, tr = _release(sim, rest_pos + [0, 0, 0.001], q, 1.0, trace_every=20)
        moved, rot = float(np.linalg.norm(p1 - rest_pos) * 1000), rot_deg(quat2mat(q1), R)
        rock = max(rot_deg(R, quat2mat(t[3:])) for t in tr[len(tr) // 2:])
        # (b) robustness: released tilted by 0-2 deg about a random horizontal axis
        ok_pert, worst = 0, 0.0
        for s in range(20):
            rng = np.random.default_rng(1000 + s)
            tilt, ax = np.radians(rng.uniform(0, 2.0)), rng.uniform(0, 2 * np.pi)
            Rp = Rotation.from_rotvec(tilt * np.array([np.cos(ax), np.sin(ax), 0])).as_matrix() @ R
            pp = rest_pos.copy()
            pp[2] = -(geo.hull_v @ Rp.T)[:, 2].min() + 0.001 + (rest_pos[2] - z)
            p2, q2, _ = _release(sim, pp, mat2quat(Rp), 1.0)
            mv, rt = np.linalg.norm(p2 - rest_pos) * 1000, rot_deg(quat2mat(q2), R)
            ok_pert += (mv < MOVE_MM) and (rt < MOVE_DEG)
            worst = max(worst, rt)
        upz = float(up @ [0, 0, 1.0])
        if NAME == "fork_stl":
            desc = ("face down: tines curve down onto the table" if upz < -0.9 else
                    "face up: lying on its back, tines curving up" if upz > 0.9 else
                    f"on its side edge ({'+' if up[1] > 0 else '-'}y side up)" if abs(upz) < 0.3 else "other")
        else:                                           # flat objects: STL +z is the top face of the plate
            desc = ("lying flat, STL +z face down" if upz < -0.9 else "lying flat, STL +z face up" if upz > 0.9 else
                    f"on its side edge ({'+' if up[1] > 0 else '-'}y side up)" if abs(upz) < 0.3 else "other")
        poses.append(dict(id=k, description=desc, quat_wxyz=q.tolist(), rpy=mat2rpy(R).tolist(),
                          rest_pos_m=rest_pos.tolist(), up_in_fork=up.tolist(),
                          trimesh_probability=c["prob"], height_mm=float((Vh[:, 2].max()) * 1000),
                          com_height_mm=float(com[2] * 1000),
                          visual_gap_um=float(((geo.V @ R.T)[:, 2].min() + z) * 1e6),
                          support_parts=sup_parts, support_points=int(len(sup)),
                          com_margin_mm=margin * 1000, tip_angle_deg=float(np.degrees(np.arctan2(margin, com[2]))),
                          release_1mm=dict(moved_mm=moved, rot_deg=rot, max_rot_last_half_s_deg=rock,
                                           passed=bool(moved < MOVE_MM and rot < MOVE_DEG)),
                          release_tilted=dict(n=20, passed=int(ok_pert), worst_rot_deg=worst)))
        print(f"pose {k}: {desc}; trimesh p={c['prob']:.3f}; 1 mm release: moved {moved:.3f} mm, {rot:.3f} deg "
              f"(rocking up to {rock:.2f} deg in the last 0.5 s); tilted releases passed {ok_pert}/20", flush=True)
    # (c) random drops
    seeds = list(range(n_drops))
    W = max(1, min(workers, n_drops))
    with get_context("spawn").Pool(W) as pool:
        obj = None if NAME == "fork_stl" and SCENE_XML is None else (NAME, SCENE_XML)
        drops = sum(pool.starmap(_drop_worker, [(seeds[i::W], obj) for i in range(W)]), [])
    U = np.array([p["up_in_fork"] for p in poses])
    clusters = {}
    for r in drops:
        ang = np.degrees(np.arccos(np.clip(U @ r["up_in_fork"], -1, 1)))
        r["pose"] = int(np.argmin(ang)) if ang.min() < 10 else -1
        r["angle_to_pose_deg"] = float(ang.min())
        clusters.setdefault(r["pose"], []).append(r)
    # unmatched drops: cluster among themselves (greedy, 10 deg)
    extra = []
    for r in clusters.get(-1, []):
        for e in extra:
            if np.degrees(np.arccos(np.clip(np.dot(e["up"], r["up_in_fork"]), -1, 1))) < 10:
                e["n"] += 1
                break
        else:
            extra.append(dict(up=r["up_in_fork"], n=1))
    for p in poses:
        hits = clusters.get(p["id"], [])
        p["drops"] = dict(n=len(hits), frequency=len(hits) / n_drops,
                          rock_deg_median=float(np.median([h["rock_deg"] for h in hits])) if hits else None)
        natural = (p["release_1mm"]["passed"] and p["release_tilted"]["passed"] >= 19
                   and p["drops"]["frequency"] >= 0.01)
        side = abs(p["up_in_fork"][2]) < 0.3
        p["category"] = "natural" if natural else ("propped" if side else "rejected")
    nat = [p for p in poses if p["category"] == "natural"]
    for p in poses:
        tot = sum(q["trimesh_probability"] for q in poses if q["category"] == p["category"])
        p["probability_within_category"] = p["trimesh_probability"] / tot if tot else 0.0
    # sampling points on the upward-facing surface, per pose
    pts = {}
    for p in poses:
        P, part = geo.top_points(quat2mat(p["quat_wxyz"]))
        pts[f"points_{p['id']}"], pts[f"part_{p['id']}"] = P, part
        p["top_surface_points"] = {PARTS[i]: int((part == i).sum()) for i in range(len(PARTS))}
    np.savez_compressed(os.path.join(out_dir, "stable_poses.npz"), **pts)
    res = dict(method=dict(
        candidates="trimesh.poses.compute_stable_poses (convex hull, quasi-static probabilities), "
                   "yaw-aligned so fork +x = world +x",
        release_1mm=f"free fork released 1 mm above the pose, 1 s; passed if moved < {MOVE_MM} mm and < {MOVE_DEG} deg",
        release_tilted="20 releases tilted 0-2 deg about random horizontal axes (+1 mm), 1 s, same thresholds",
        drops=f"{n_drops} random orientations (uniform), lowest point 2-10 mm above the table, 3 s; final "
              "orientation matched to a candidate by its up vector (< 10 deg)",
        natural="release_1mm passed AND >= 19/20 tilted releases passed AND drop frequency >= 1 %",
        propped="side-edge pose that is not natural: kept only as a separate category (fork held by a fixture)",
        sim="GelSightSim(free_objects=True): CoACD parts vs the table plane, the simulator's own contact "
            "settings (incl. the free object's invweight0 = 5 % of the sensor's)"),
        poses=poses, drops=dict(n=n_drops, unmatched=len(clusters.get(-1, [])), unmatched_clusters=extra,
                                per_pose={p["id"]: p["drops"]["n"] for p in poses}),
        elapsed_s=time.time() - t0)
    with open(os.path.join(out_dir, "stable_poses.json"), "w") as f:
        json.dump(res, f, indent=1)
    pose_figure(poses, os.path.join(out_dir, "stable_poses.png"))
    print(f"natural: {[p['id'] for p in nat]}, propped: {[p['id'] for p in poses if p['category'] == 'propped']}, "
          f"rejected: {[p['id'] for p in poses if p['category'] == 'rejected']}; drops per pose "
          f"{res['drops']['per_pose']}, unmatched {res['drops']['unmatched']} ({time.time() - t0:.0f} s)")
    return res


def pose_figure(poses, path):
    from gelsight_sim import GelSightSim
    sim = GelSightSim(xml_path=SCENE_XML, mode="moving_sensor", seed=0)
    tiles = []
    for p in poses:
        sim.set_rest_pose(NAME, p["rest_pos_m"], p["quat_wxyz"])
        sim.place_object(NAME)
        sim.set_pose(x=0.15, y=0.15, z=0.08)
        R, t = quat2mat(p["quat_wxyz"]), np.array(p["rest_pos_m"])
        if NAME == "fork_stl":
            head = R @ np.array([0.024, 0, 0.0025]) + t      # middle of the tines, world
            tail = R @ np.array([-0.072, 0, 0.0008]) + t     # near the handle end
        else:                                               # 12 mm in from either end, mid-height
            bb = np.array(_objects.get(NAME)["bbox_mm"]) / 1000.0
            zm = bb[:, 2].mean()
            head = R @ np.array([bb[1, 0] - 0.012, 0, zm]) + t
            tail = R @ np.array([bb[0, 0] + 0.012, 0, zm]) + t
        head[2] = tail[2] = 0.004
        vw = dict(width=400, height=260)
        views = [sim.render_scene(azimuth=-60, elevation=-28, distance=0.17, lookat=(0, 0, 0.004), **vw),
                 sim.render_scene(azimuth=180, elevation=-12, distance=0.22, lookat=(0, 0, 0.006), **vw),
                 sim.render_scene(azimuth=-90, elevation=0, distance=0.05, lookat=head, **vw),
                 sim.render_scene(azimuth=-90, elevation=0, distance=0.05, lookat=tail, **vw)]
        row = np.concatenate([v[..., ::-1] for v in views], 1)
        bar = np.zeros((44, row.shape[1], 3), np.uint8)
        t1 = f"pose {p['id']} [{p['category'].upper()}]  {p['description']}"
        t2 = (f"trimesh p={p['trimesh_probability']:.3f}  drops {p['drops']['frequency'] * 100:.1f}%  "
              f"1mm release: {p['release_1mm']['moved_mm']:.3f} mm / {p['release_1mm']['rot_deg']:.2f} deg  "
              f"tilted releases {p['release_tilted']['passed']}/20  tip angle {p['tip_angle_deg']:.1f} deg  "
              f"rests on {', '.join(p['support_parts'])}  height {p['height_mm']:.1f} mm")
        cv2.putText(bar, t1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, t2, (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([bar, row], 0))
    head = np.zeros((26, tiles[0].shape[1], 3), np.uint8)
    cv2.putText(head, f"{NAME} stable poses: oblique | end view from the {'tine tips' if NAME == 'fork_stl' else 'head'} | "
                f"side close-up of the {'tines' if NAME == 'fork_stl' else 'head'} | "
                "side close-up of the handle end (side views look along world -y: tips to the LEFT)",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, np.concatenate([head] + tiles, 0))
    sim.close()


# ================================================================ PART 2+3: samples
class Collector:
    """Generates samples: a static (welded) press for the images and labels, and the same
    press with a free fork for would_move."""

    def __init__(self, poses_json, poses_npz, cfg: Config):
        import mujoco
        from gelsight_sim import GelSightSim, MOVING_XML, sensor_model
        from fork_detector import ForkDetector
        self.cfg = cfg
        set_sensor(cfg.sensor)
        self.geo = ForkGeometry()
        self.sim = GelSightSim(xml_path=SCENE_XML, mode="moving_sensor", seed=0, render_table=True, sensor=cfg.sensor)
        self.free = GelSightSim(xml_path=SCENE_XML, mode="moving_sensor", free_objects=True, seed=0, sensor=cfg.sensor)
        if not np.isclose(self.sim.mm_per_px, 0.1) or len(self.sim._markers) != N_MARKERS:
            raise RuntimeError("sensor image size / markers disagree with set_sensor()")
        for s in (self.sim, self.free):
            s.place_object(NAME)
        self.det = ForkDetector.load(DETECTOR)
        m = self.sim.model
        self.fb = m.body(NAME).id
        self.gel, self.table = self.sim.gel_id, self.sim.table_id
        # fork collision geoms -> (CoACD part index, tine index)
        self.coacd = {}
        for g in self.sim.collision_geoms:
            nm = m.geom(g).name
            if nm.startswith(NAME + "_c"):
                tine = int(nm.split("_tine")[1]) if "_tine" in nm else -1
                self.coacd[g] = (int(nm[len(NAME) + 2:len(NAME) + 4]), tine)
        self.coacd_free = {}
        for g in self.free.collision_geoms:
            nm = self.free.model.geom(g).name
            if nm.startswith(NAME + "_c"):
                tine = int(nm.split("_tine")[1]) if "_tine" in nm else -1
                self.coacd_free[g] = (int(nm[len(NAME) + 2:len(NAME) + 4]), tine)
        _set_free_dims(self.free.model.nq, self.free.model.nv)
        self.gel_half = m.geom_size[self.gel][:2].copy()
        self.mass = float(m.body_mass[self.fb])
        # poses used for sampling
        self.poses = [p for p in poses_json["poses"] if p["category"] in ("natural", "propped")]
        nat = [p for p in self.poses if p["category"] == "natural"]
        prop = [p for p in self.poses if p["category"] == "propped"]
        pf = cfg.propped_frac if prop else 0.0
        w = []
        for p in self.poses:
            group = nat if p["category"] == "natural" else prop
            share = (1 - pf) if p["category"] == "natural" else pf
            w.append(share * p["trimesh_probability"] / sum(q["trimesh_probability"] for q in group))
        self.pose_w = np.array(w) / np.sum(w)
        self.pts = {p["id"]: (poses_npz[f"points_{p['id']}"], poses_npz[f"part_{p['id']}"]) for p in self.poses}
        self.hull2d = {}
        from scipy.spatial import ConvexHull
        for p in self.poses:
            R = quat2mat(p["quat_wxyz"])
            V2 = (self.geo.V @ R.T)[:, :2] - (R @ self.geo.bbox_centre)[:2]      # relative to the bbox centre
            self.hull2d[p["id"]] = V2[ConvexHull(V2).vertices]
        self.model_hash = model_hash(mujoco.MjModel.from_xml_path(SCENE_XML or MOVING_XML) if cfg.sensor == "standard"
                                     else sensor_model(cfg.sensor, SCENE_XML))

    # ------------------------------------------------------------ placement
    def _place(self, pose, cx, cy, yaw):
        import mujoco
        for s in (self.sim, self.free):
            mujoco.mj_resetDataKeyframe(s.model, s.data, 0)
            s.set_rest_pose(NAME, pose["rest_pos_m"], pose["quat_wxyz"])
            s.place_object(NAME, x=cx, y=cy, yaw=yaw)

    def _draw_sensor(self, rng, stype, pose, cx, cy, fyaw):
        c = self.cfg
        yaw = rng.uniform(-np.pi, np.pi)
        th, ph = np.radians(c.tilt_max_deg) * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
        pitch, roll = th * np.cos(ph), th * np.sin(ph)
        S = arm_axes(yaw, pitch, roll)
        info = dict(target_part=-1, target_w=np.full(3, np.nan), offset_s=np.full(2, np.nan), fail=0)
        if stype in ("on_fork", "hover"):
            P, part = self.pts[pose["id"]]
            avail = [i for i in range(len(PARTS)) if (part == i).sum() >= 50 and c.part_weights.get(PARTS[i], 0) > 0]
            pw = np.array([c.part_weights[PARTS[i]] for i in avail], float)
            k = avail[rng.choice(len(avail), p=pw / pw.sum())]
            idx = np.nonzero(part == k)[0]
            pb = P[idx[rng.integers(len(idx))]] / 1000.0
            d = self.sim.data
            tw = d.xpos[self.fb] + d.xmat[self.fb].reshape(3, 3) @ pb
            off = rng.uniform(-1, 1, 2) * self.gel_half
            cxy = tw[:2] - S[:2, :2] @ off
            info.update(target_part=k, target_w=tw, offset_s=off)
        else:
            poly = self.hull2d[pose["id"]] @ rz(fyaw)[:2, :2].T + [cx, cy]
            half = self.gel_half + c.table_margin_mm / 1000.0
            ax = S[:2, :2].T                                  # rows: sensor x, y directions (2-D)
            ax = ax / np.linalg.norm(ax, axis=1, keepdims=True)
            for _ in range(200):
                r, a = c.table_radius_m * np.sqrt(rng.uniform()), rng.uniform(0, 2 * np.pi)
                cxy = np.array([cx + r * np.cos(a), cy + r * np.sin(a)])
                if not rect_hits_polygon(cxy, ax, half, poly):
                    break
            else:
                info["fail"] = RETRY["no_free_spot"]
        return dict(x=float(cxy[0]), y=float(cxy[1]), yaw=float(yaw), pitch=float(pitch), roll=float(roll)), info

    # ------------------------------------------------------------ contacts
    def _gel_contacts(self, o, Rs, Rf, pf):
        import mujoco
        m, d = self.sim.model, self.sim.data
        rows = []
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if self.gel not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.gel else c.geom1
            sign = 1.0 if c.geom2 == self.gel else -1.0
            mujoco.mj_contactForce(m, d, i, f6)
            fr = c.frame.reshape(3, 3)
            n_w = sign * fr[0]                              # from the other geom into the gel
            f_w = sign * (fr.T @ f6[:3])                    # force on the gel
            kind = 0 if other in self.coacd else (1 if other == self.table else 2)
            part, cp, tine = -1, -1, -1
            if kind == 0:
                x_f = Rf.T @ (c.pos - pf)
                part = int(part_of(x_f[0] * 1000, (Rf.T @ n_w)[2]))
                cp, tine = self.coacd[other]
            rows.append((Rs.T @ (c.pos - o), c.pos.copy(), Rs.T @ n_w, f6[0], f6[1:3].copy(), Rs.T @ f_w,
                         c.dist, kind, part, cp, tine))
        return rows

    def _table_contacts(self):
        import mujoco
        m, d = self.free.model, self.free.data
        b = self.free.tool_body[NAME]
        Rf, pf = d.xmat[b].reshape(3, 3), d.xpos[b]
        rows = []
        f6 = np.zeros(6)
        F_table = np.zeros(3)
        F_gel = np.zeros(3)
        for i in range(d.ncon):
            c = d.contact[i]
            gs = (c.geom1, c.geom2)
            fork_g = [g for g in gs if g in self.coacd_free]
            if not fork_g:
                continue
            other = c.geom2 if c.geom1 == fork_g[0] else c.geom1
            sign = 1.0 if c.geom2 == fork_g[0] else -1.0     # orient towards the fork
            mujoco.mj_contactForce(m, d, i, f6)
            fr = c.frame.reshape(3, 3)
            f_on_fork = sign * (fr.T @ f6[:3])
            if other == self.free.table_id:
                F_table += f_on_fork
                n_w = sign * fr[0]
                x_f = Rf.T @ (c.pos - pf)
                cp, tine = self.coacd_free[fork_g[0]]
                rows.append((c.pos.copy(), n_w, f6[0], f6[1:3].copy(), c.dist,
                             int(part_of(x_f[0] * 1000, (Rf.T @ n_w)[2])), cp, tine))
            elif other in self.free.sensor_geoms:
                F_gel += f_on_fork
        resid = F_table + F_gel + self.mass * self.free.model.opt.gravity
        return rows, float(np.linalg.norm(resid))

    # ------------------------------------------------------------ one sample
    def sample(self, seed, ref):
        import mujoco
        c, sim, fr = self.cfg, self.sim, self.free
        m, d = sim.model, sim.data
        rng = np.random.default_rng(seed)
        k = int(rng.choice(len(self.poses), p=self.pose_w))
        pose = self.poses[k]
        cx, cy = rng.uniform(-c.fork_xy_range_m, c.fork_xy_range_m, 2)
        fyaw = rng.uniform(-np.pi, np.pi)
        fr_t = np.array([c.frac_on_fork, c.frac_table, c.frac_hover])
        stype = SAMPLE_TYPES[int(rng.choice(3, p=fr_t / fr_t.sum()))]
        force = float(np.exp(rng.uniform(np.log(c.force_min_n), np.log(c.force_max_n))))
        gap = float(rng.uniform(*c.hover_gap_mm)) / 1000.0
        reasons, tries = 0, 0
        while True:
            self._place(pose, cx, cy, fyaw)
            arm, tinfo = self._draw_sensor(rng, stype, pose, cx, cy, fyaw)
            fail = tinfo["fail"]
            if stype == "hover":
                z_c = sim.contact_height(**arm)
                sim.set_pose(z=z_c + gap, **arm)
                settled = True
            else:
                info = sim.press(force, max_time=c.press_max_s, **arm)
                settled = info.settled
                fail |= RETRY["housing"] * info.housing_contact | RETRY["unsettled"] * (not settled)
                if stype == "table":
                    touch_fork = any(self.gel in (d.contact[i].geom1, d.contact[i].geom2) and
                                     (d.contact[i].geom1 in self.coacd or d.contact[i].geom2 in self.coacd)
                                     for i in range(d.ncon))
                    fail |= RETRY["fork_touch"] * touch_fork
            if not fail or tries + 1 >= c.max_tries:
                break
            reasons |= fail
            tries += 1
        flawed = bool(fail)
        reasons |= fail
        # ---- static run: state, frames, images
        o, Rs = sim.sensor_frame()
        Rf, pf = d.xmat[self.fb].reshape(3, 3).copy(), d.xpos[self.fb].copy()
        com_w, bbox_w = d.xipos[self.fb].copy(), pf + Rf @ self.geo.bbox_centre
        h_fork = sim.indentation_mm(raw=True, table=False)
        h_raw = sim.indentation_mm(raw=True, table=True)
        h_gel = sim.gel_surface(h_raw)
        sim.rng = np.random.default_rng([seed, 1])       # image noise: also from the seed
        img = sim.shade(h_gel)
        rec = self.det.reconstruct(img, ref)
        _, shift = sim.marker_field(h_gel)
        gel_c = self._gel_contacts(o, Rs, Rf, pf)
        touch = float(d.sensordata[sim.touch_adr])
        a2 = sim.mm_per_px ** 2
        s = dict(seed=seed, sample_type=SAMPLE_TYPES.index(stype), pose_id=pose["id"],
                 pose_category=CATEGORIES.index(pose["category"]), target_part=tinfo["target_part"],
                 target_w=tinfo["target_w"], target_offset_s=tinfo["offset_s"],
                 fork_place=np.array([cx, cy, fyaw]),
                 arm_cmd=np.array([arm["x"], arm["y"], arm["yaw"], arm["pitch"], arm["roll"]]),
                 force_cmd=0.0 if stype == "hover" else force, hover_gap=gap if stype == "hover" else np.nan,
                 force_touch=touch, peak_indent_mm=float(h_raw.max()), peak_fork_mm=float(h_fork.max()),
                 contact_fork=any(r[7] == 0 for r in gel_c), contact_table=any(r[7] == 1 for r in gel_c),
                 fork_visible=bool(((h_fork > 0.05).sum() * a2) > 0.5),
                 contact_visible=bool(((h_raw > 0.05).sum() * a2) > 0.5),
                 saturated=bool(h_raw.max() >= sim.opt.max_indent_mm - 1e-6),
                 n_tines=sim.visible_tines(), settled=settled, retries=tries, retry_reasons=reasons, flawed=flawed,
                 qpos=d.qpos.copy(), qvel=d.qvel.copy(), ctrl=d.ctrl.copy(),
                 qacc_warmstart=d.qacc_warmstart.copy(),
                 fork_body_pos=m.body_pos[self.fb].copy(), fork_body_quat=m.body_quat[self.fb].copy(),
                 fork_pos_w=pf, fork_quat_w=mat2quat(Rf), fork_rpy_w=mat2rpy(Rf),
                 sensor_pos_w=o, sensor_quat_w=mat2quat(Rs), sensor_rpy_w=mat2rpy(Rs),
                 fork_pos_s=Rs.T @ (pf - o), fork_quat_s=mat2quat(Rs.T @ Rf), fork_rpy_s=mat2rpy(Rs.T @ Rf),
                 com_s=Rs.T @ (com_w - o), bbox_s=Rs.T @ (bbox_w - o), com_w=com_w, bbox_w=bbox_w,
                 tilt_deg=float(np.degrees(np.arccos(np.clip(-Rs[2, 2], -1, 1)))),
                 gel_force_s=np.sum([r[5] for r in gel_c], 0) if gel_c else np.zeros(3),
                 img=img, depth_gel=h_gel.astype(np.float16), depth_fork=h_fork.astype(np.float16),
                 depth_recon=rec.astype(np.float16), marker_shift=shift.astype(np.float32))
        s["gel_contacts"] = gel_c
        if not c.free_run:
            # the fork is clamped to the table: no free-fork run; the free-run fields are placeholders
            s.update(would_move=False, free_shift_mm=np.nan, free_origin_shift_mm=np.nan, free_rot_deg=np.nan,
                     free_settled=False, free_force_touch=np.nan, free_force_residual=np.nan,
                     free_fork_pos_w=np.full(3, np.nan), free_fork_quat_w=np.full(4, np.nan),
                     free_qpos=np.full(fr.model.nq, np.nan), free_qvel=np.full(fr.model.nv, np.nan))
            s["table_contacts"] = []
            return s
        # ---- free run: same press (or hover) with the fork on a freejoint
        fm, fd = fr.model, fr.data
        b = fr.tool_body[NAME]
        com0, R0, p0 = fd.xipos[b].copy(), fd.xmat[b].reshape(3, 3).copy(), fd.xpos[b].copy()
        if stype == "hover":
            fr.set_pose(z=d.qpos[sim.jnt["slide_z"]], **arm)
            for _ in range(int(round(c.free_hold_s / fm.opt.timestep))):
                mujoco.mj_step(fm, fd)
            mujoco.mj_forward(fm, fd)
            f_settled, f_touch = True, float(fd.sensordata[fr.touch_adr])
        else:
            finfo = fr.press(force, max_time=c.press_max_s, **arm)
            f_settled, f_touch = finfo.settled, finfo.gel_force
        shift_mm = float(np.linalg.norm(fd.xipos[b] - com0) * 1000)
        rot = rot_deg(R0, fd.xmat[b].reshape(3, 3))
        tab, resid = self._table_contacts()
        s.update(would_move=bool(shift_mm > MOVE_MM or rot > MOVE_DEG), free_shift_mm=shift_mm,
                 free_origin_shift_mm=float(np.linalg.norm(fd.xpos[b] - p0) * 1000), free_rot_deg=rot,
                 free_settled=bool(f_settled), free_force_touch=f_touch, free_force_residual=resid,
                 free_fork_pos_w=fd.xpos[b].copy(), free_fork_quat_w=fd.xquat[b].copy(),
                 free_qpos=fd.qpos.copy(), free_qvel=fd.qvel.copy())
        s["table_contacts"] = tab
        return s


# ---------------------------------------------------------------- HDF5 layout
H, W = 240, 320
N_MARKERS = 192
FIELDS = {   # name: (per-sample shape, dtype)
    "seed": ((), np.int64), "sample_type": ((), np.uint8), "pose_id": ((), np.uint8),
    "pose_category": ((), np.uint8), "target_part": ((), np.int8), "target_w": ((3,), np.float64),
    "target_offset_s": ((2,), np.float64), "fork_place": ((3,), np.float64), "arm_cmd": ((5,), np.float64),
    "force_cmd": ((), np.float64), "hover_gap": ((), np.float64), "force_touch": ((), np.float64),
    "peak_indent_mm": ((), np.float32), "peak_fork_mm": ((), np.float32), "contact_fork": ((), bool),
    "contact_table": ((), bool), "fork_visible": ((), bool), "contact_visible": ((), bool),
    "saturated": ((), bool), "n_tines": ((), np.uint8), "settled": ((), bool), "retries": ((), np.uint8),
    "retry_reasons": ((), np.uint8), "flawed": ((), bool),
    "qpos": ((6,), np.float64), "qvel": ((6,), np.float64), "ctrl": ((6,), np.float64),
    "qacc_warmstart": ((6,), np.float64), "fork_body_pos": ((3,), np.float64), "fork_body_quat": ((4,), np.float64),
    "fork_pos_w": ((3,), np.float64), "fork_quat_w": ((4,), np.float64), "fork_rpy_w": ((3,), np.float64),
    "sensor_pos_w": ((3,), np.float64), "sensor_quat_w": ((4,), np.float64), "sensor_rpy_w": ((3,), np.float64),
    "fork_pos_s": ((3,), np.float64), "fork_quat_s": ((4,), np.float64), "fork_rpy_s": ((3,), np.float64),
    "com_s": ((3,), np.float64), "bbox_s": ((3,), np.float64), "com_w": ((3,), np.float64),
    "bbox_w": ((3,), np.float64), "tilt_deg": ((), np.float32), "gel_force_s": ((3,), np.float64),
    "would_move": ((), bool), "free_shift_mm": ((), np.float32), "free_origin_shift_mm": ((), np.float32),
    "free_rot_deg": ((), np.float32), "free_settled": ((), bool), "free_force_touch": ((), np.float64),
    "free_force_residual": ((), np.float64), "free_fork_pos_w": ((3,), np.float64),
    "free_fork_quat_w": ((4,), np.float64), "free_qpos": ((13,), np.float64), "free_qvel": ((12,), np.float64),
    "img": ((H, W, 3), np.uint8), "depth_gel": ((H, W), np.float16), "depth_fork": ((H, W), np.float16),
    "depth_recon": ((H, W), np.float16), "marker_shift": ((N_MARKERS, 2), np.float32),
}
def set_sensor(name):
    """Image size and marker count of the sensor (module globals H, W, N_MARKERS and the image FIELDS);
    also exported as GELSIGHT_SENSOR so spawned workers and GelSightSim(sensor=None) agree."""
    global H, W, N_MARKERS
    from gelsight_sim import SENSORS, SENSOR_ENV, OpticsConfig
    prof = SENSORS[name]
    H, W = prof["height"], prof["width"]
    pitch = OpticsConfig().marker_pitch_mm / 0.1                      # 0.1 mm/px for every sensor
    N_MARKERS = int(W / pitch) * int(H / pitch)
    FIELDS.update({"img": ((H, W, 3), np.uint8), "depth_gel": ((H, W), np.float16),
                   "depth_fork": ((H, W), np.float16), "depth_recon": ((H, W), np.float16),
                   "marker_shift": ((N_MARKERS, 2), np.float32)})
    os.environ[SENSOR_ENV] = name


def _set_free_dims(nq, nv):
    """The free model has a freejoint on every object (16 x 7 / 6) besides the arm's 6 joints."""
    FIELDS["free_qpos"] = ((int(nq),), np.float64)
    FIELDS["free_qvel"] = ((int(nv),), np.float64)


GEL_C = [("pos_s", (3,), np.float64), ("pos_w", (3,), np.float64), ("normal_s", (3,), np.float64),
         ("fn", (), np.float64), ("ft", (2,), np.float64), ("force_s", (3,), np.float64), ("dist", (), np.float64),
         ("other", (), np.int8), ("part", (), np.int8), ("coacd_part", (), np.int8), ("tine", (), np.int8)]
TAB_C = [("pos_w", (3,), np.float64), ("normal_w", (3,), np.float64), ("fn", (), np.float64),
         ("ft", (2,), np.float64), ("dist", (), np.float64), ("part", (), np.int8), ("coacd_part", (), np.int8),
         ("tine", (), np.int8)]
IMAGE_FIELDS = ("img", "depth_gel", "depth_fork", "depth_recon")


def metadata(col: Collector, poses_json, cfg: Config):
    import mujoco
    sim = col.sim
    m = sim.model
    b, g = col.fb, sim.gel_id
    Ri = quat2mat(m.body_iquat[b])
    det = col.det
    return dict(
        description="GelSight tactile dataset of fork_stl (ForkMilimeter.stl), moving sensor; see README.md",
        conventions=CONVENTIONS, units_note="see conventions",
        fields={k: dict(shape=list(s), dtype=np.dtype(t).name) for k, (s, t) in FIELDS.items()},
        gel_contact_fields={k: dict(shape=list(s), dtype=np.dtype(t).name) for k, s, t in GEL_C},
        table_contact_fields={k: dict(shape=list(s), dtype=np.dtype(t).name) for k, s, t in TAB_C},
        codes=dict(sample_type=SAMPLE_TYPES, pose_category=CATEGORIES, part=PARTS, retry_reasons=RETRY,
                   gel_contact_other=["fork", "table", "other"]),
        fork=dict(mass_kg=float(m.body_mass[b]), com_f=m.body_ipos[b].tolist(),
                  inertia_principal=m.body_inertia[b].tolist(), inertia_quat_wxyz=m.body_iquat[b].tolist(),
                  inertia_tensor_f=(Ri @ np.diag(m.body_inertia[b]) @ Ri.T).tolist(),
                  bbox_centre_f=col.geo.bbox_centre.tolist(), bbox_f=(col.geo.vis.bounds / 1000).tolist(),
                  mesh="meshes/fork_stl/fork_stl_visual.stl (mm, scale 0.001)", coacd_parts=col.geo.part_names,
                  part_boundaries_mm=(dict(tine_root=TINE_ROOT_MM, palm_start=PALM_START_MM,
                                           neck_start=NECK_START_MM, edge_abs_nz_below=EDGE_NZ) if NAME == "fork_stl"
                                      else dict(names=PARTS[:-1], x_starts_mm=X_STARTS_MM, edge_abs_nz_below=EDGE_NZ))),
        gel=dict(size_half_m=m.geom_size[g].tolist(), solref=m.geom_solref[g].tolist(),
                 solimp=m.geom_solimp[g].tolist(), friction=m.geom_friction[g].tolist(),
                 priority=int(m.geom_priority[g]), condim=int(m.geom_condim[g]),
                 touch_site_half_m=m.site("gel_touch").size.tolist(), d0_m=sim.d0),
        sim=dict(timestep=m.opt.timestep, integrator=int(m.opt.integrator), cone=int(m.opt.cone),
                 solver=int(m.opt.solver), iterations=int(m.opt.iterations), gravity=m.opt.gravity.tolist(),
                 tactile_render="fork visual mesh + table (render_table=True)",
                 static_fork="welded: body without joints (fixed to the world)",
                 free_fork="freejoint, GelSightSim(free_objects=True)"),
        optics=asdict(sim.opt), image=dict(height=H, width=W, mm_per_px=sim.mm_per_px),
        marker_rest_px=sim._markers.tolist(),
        photometric_calibration=dict(model=os.path.relpath(DETECTOR, HERE), A=det.pm.A.tolist(),
                                     b=np.asarray(det.pm.b).tolist(),
                                     pipeline="fork_detector.PhotometricModel.height_map (markers removed, "
                                              "photometric stereo, Poisson/DCT integration)"),
        config=asdict(cfg), poses=poses_json["poses"],
        seeds=dict(sample=f"split_index * {SEED_BASE} + index (splits {SPLITS})",
                   reference_image=f"{REF_SEED_BASE} + 1000 * split_index + shard_index",
                   image_noise="np.random.default_rng([seed, 1])"),
        provenance=dict(git_commit=git_commit(), git_note=None if git_commit() else "not a git repository",
                        source_sha256=source_hashes(), mesh_sha256=mesh_hash(), model_sha256=col.model_hash,
                        mujoco=mujoco.__version__, numpy=np.__version__, python=platform.python_version(),
                        host=platform.node()))


def _seed(split, i):
    return SPLITS.index(split) * SEED_BASE + i


def write_shard(args):
    path, split, shard, idx, cfg_d, poses_path = args
    import h5py
    cfg = Config(**cfg_d)
    with open(poses_path) as f:
        poses_json = json.load(f)
    poses_npz = np.load(poses_path.replace(".json", ".npz"))
    col = Collector(poses_json, poses_npz, cfg)
    ref_seed = REF_SEED_BASE + 1000 * SPLITS.index(split) + shard
    col.sim.rng = np.random.default_rng(ref_seed)
    ref = col.sim.reference_image()
    n = len(idx)
    tmp = path + ".part"
    t0 = time.time()
    gel_rows, tab_rows = [], []
    gel_cnt, tab_cnt = np.zeros(n, np.int64), np.zeros(n, np.int64)
    with h5py.File(tmp, "w") as f:
        meta = metadata(col, poses_json, cfg)
        f.create_dataset("metadata_json", data=json.dumps(meta))
        f.attrs.update(split=split, shard=shard, n=n, ref_seed=ref_seed, first_seed=_seed(split, idx[0]))
        f["reference_image"] = ref
        f["marker_rest"] = col.sim._markers.astype(np.float32)
        ds = {}
        for k, (shp, dt) in FIELDS.items():
            if k in IMAGE_FIELDS:
                ds[k] = f.create_dataset(k, (n,) + shp, dtype=dt, chunks=(1,) + shp, compression="gzip",
                                         compression_opts=4, shuffle=True)
            else:
                ds[k] = f.create_dataset(k, (n,) + shp, dtype=dt)
        t_s = []
        for j, i in enumerate(idx):
            ts = time.time()
            s = col.sample(_seed(split, i), ref)
            for k in FIELDS:
                ds[k][j] = s[k]
            gel_rows += s["gel_contacts"]
            tab_rows += s["table_contacts"]
            gel_cnt[j], tab_cnt[j] = len(s["gel_contacts"]), len(s["table_contacts"])
            t_s.append(time.time() - ts)
            if (j + 1) % 100 == 0:
                print(f"  {split}-{shard:04d}: {j + 1}/{n} ({np.mean(t_s[-100:]):.3f} s/sample)", flush=True)
        for grp, spec, rows, cnt in (("gel_contacts", GEL_C, gel_rows, gel_cnt),
                                     ("table_contacts", TAB_C, tab_rows, tab_cnt)):
            gp = f.create_group(grp)
            gp["count"] = cnt
            gp["start"] = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
            for c, (name, shp, dt) in enumerate(spec):
                arr = np.array([r[c] for r in rows], dtype=dt).reshape((len(rows),) + shp)
                gp.create_dataset(name, data=arr, compression="gzip", compression_opts=4, shuffle=True)
        f.attrs["seconds"] = time.time() - t0
        f.attrs["seconds_per_sample"] = float(np.mean(t_s))
    os.replace(tmp, path)
    return dict(path=path, n=n, seconds=time.time() - t0, bytes=os.path.getsize(path),
                s_per_sample=float(np.mean(t_s)))


def collect(out, n, splits, shard_size, workers, cfg: Config, only_split=None):
    from multiprocessing import get_context
    poses_path = os.path.join(out, "poses", "stable_poses.json")
    if not os.path.exists(poses_path):
        raise SystemExit(f"{poses_path} missing: run `python collect_dataset.py poses` first")
    if only_split:                                  # all n samples into one split (its own seed range)
        sizes = [n if s_ == only_split else 0 for s_ in SPLITS]
    else:
        sizes = [int(round(n * f)) for f in splits] + [0] * (len(SPLITS) - 3)
        sizes[0] = n - sum(sizes[1:])
    tasks = []
    for split, sz in zip(SPLITS, sizes):
        for shard in range(int(np.ceil(sz / shard_size))):
            idx = list(range(shard * shard_size, min(sz, (shard + 1) * shard_size)))
            path = os.path.join(out, f"{split}-{shard:04d}.h5")
            if os.path.exists(path):
                print("exists, skipped:", path)
                continue
            tasks.append((path, split, shard, idx, asdict(cfg), poses_path))
    t0 = time.time()
    print(f"{len(tasks)} shards, {sum(len(t[3]) for t in tasks)} samples, {min(workers, max(len(tasks), 1))} workers",
          flush=True)
    res = []
    if tasks:
        with get_context("spawn").Pool(min(workers, len(tasks))) as pool:
            for r in pool.imap_unordered(write_shard, tasks):
                print(f"wrote {os.path.relpath(r['path'], HERE)}: {r['n']} samples, {r['seconds']:.0f} s, "
                      f"{r['bytes'] / 1e6:.0f} MB", flush=True)
                res.append(r)
    wall = time.time() - t0
    # one copy of the metadata next to the shards
    import h5py
    first = sorted(glob.glob(os.path.join(out, "*-0000.h5")))[0]
    with h5py.File(first) as f:
        meta = json.loads(f["metadata_json"][()])
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    if res:
        nn = sum(r["n"] for r in res)
        by = sum(r["bytes"] for r in res)
        spw = np.mean([r["s_per_sample"] for r in res])
        th = dict(samples=nn, wall_s=wall, workers=min(workers, len(tasks)), samples_per_s=nn / wall,
                  seconds_per_sample_per_worker=spw, bytes_per_sample=by / nn,
                  projected_100k=dict(size_gb=by / nn * 1e5 / 1e9,
                                      hours_at_this_parallelism=1e5 / (nn / wall) / 3600))
        with open(os.path.join(out, "throughput.json"), "w") as f:
            json.dump(th, f, indent=1)
        print(json.dumps(th, indent=1))


# ================================================================ reading / regeneration
def load_sample(f, i):
    s = {k: f[k][i] for k in FIELDS}
    for grp, spec in (("gel_contacts", GEL_C), ("table_contacts", TAB_C)):
        a, c = int(f[grp]["start"][i]), int(f[grp]["count"][i])
        s[grp] = {name: f[grp][name][a:a + c] for name, _, _ in spec}
    return s


def compare(stored, fresh):
    """Field-by-field equality of a stored sample and a regenerated one."""
    bad = []
    for k, (shp, dt) in FIELDS.items():
        a, b = np.asarray(stored[k]), np.asarray(fresh[k], dtype=dt)
        if not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
            bad.append(k)
    for grp, spec, key in (("gel_contacts", GEL_C, "gel_contacts"), ("table_contacts", TAB_C, "table_contacts")):
        rows = fresh[key]
        for c, (name, shp, dt) in enumerate(spec):
            b = np.array([r[c] for r in rows], dtype=dt).reshape((len(rows),) + shp)
            if not np.array_equal(stored[grp][name], b):
                bad.append(f"{grp}/{name}")
    return bad


def regenerate(path, indices):
    import h5py
    out = []
    with h5py.File(path) as f:
        meta = json.loads(f["metadata_json"][()])
        # shards written before the free_run option always had the free-fork run
        cfg = Config(**{"free_run": True, **{k: (tuple(v) if k == "hover_gap_mm" else v) for k, v in meta["config"].items()}})
        poses_json = dict(poses=meta["poses"])
        poses_npz = np.load(os.path.join(os.path.dirname(path), "poses", "stable_poses.npz"))
        col = Collector(poses_json, poses_npz, cfg)
        ref = f["reference_image"][()]
        for i in indices:
            st = load_sample(f, i)
            fresh = col.sample(int(st["seed"]), ref)
            bad = compare(st, fresh)
            out.append(dict(shard=os.path.basename(path), index=int(i), seed=int(st["seed"]), identical=not bad,
                            differing_fields=bad))
    return out, col


def replay_state(col, path, i):
    """Rebuild the static sample from its stored state (fork pose + qpos/qvel/ctrl), no simulation."""
    import h5py
    import mujoco
    sim = col.sim
    with h5py.File(path) as f:
        st = load_sample(f, i)
        ref = f["reference_image"][()]
    m, d = sim.model, sim.data
    sim.set_tool(NAME)
    m.body_pos[col.fb], m.body_quat[col.fb] = st["fork_body_pos"], st["fork_body_quat"]
    d.qpos[:], d.qvel[:], d.ctrl[:] = st["qpos"], st["qvel"], st["ctrl"]
    d.qacc_warmstart[:] = st["qacc_warmstart"]
    mujoco.mj_forward(m, d)
    h_gel = sim.gel_surface(sim.indentation_mm(raw=True, table=True))
    sim.rng = np.random.default_rng([int(st["seed"]), 1])
    img = sim.shade(h_gel)
    return dict(img_identical=bool(np.array_equal(img, st["img"])),
                depth_gel_identical=bool(np.array_equal(h_gel.astype(np.float16), st["depth_gel"])),
                recon_identical=bool(np.array_equal(col.det.reconstruct(img, ref).astype(np.float16),
                                                    st["depth_recon"])),
                touch_identical=bool(float(d.sensordata[sim.touch_adr]) == float(st["force_touch"])),
                touch_abs_diff_n=abs(float(d.sensordata[sim.touch_adr]) - float(st["force_touch"])))


# ================================================================ PART 4: verification
def label_maps(geo, st, mm_per_px):
    """From the LABELS only (fork pose in S, sensor pose in W): fork silhouette, fork penetration
    of the gel plane and table penetration, all (H, W), plus pixel positions of COM / bbox centre."""
    R_sf = quat2mat(st["fork_quat_s"])
    P = geo.V @ R_sf.T + st["fork_pos_s"]
    pix = np.c_[P[:, 0] * 1000 / mm_per_px + (W - 1) / 2, P[:, 1] * 1000 / mm_per_px + (H - 1) / 2, P[:, 2]]
    zmin, tid = rasterize(pix, geo.F, H, W, keep="min")
    pen_fork = np.where(np.isfinite(zmin), np.clip(-zmin * 1000, 0, 3.0), 0.0)
    # table plane z_W = 0 seen from the sensor: z_S of the plane under each pixel
    o, Rs = st["sensor_pos_w"], quat2mat(st["sensor_quat_w"])
    yy, xx = np.mgrid[0:H, 0:W]
    xs, ys = (xx - (W - 1) / 2) * mm_per_px / 1000, (yy - (H - 1) / 2) * mm_per_px / 1000
    zs = -(o[2] + Rs[2, 0] * xs + Rs[2, 1] * ys) / Rs[2, 2]
    pen_table = np.clip(-zs * 1000, 0, 3.0)
    px = lambda p: (p[0] * 1000 / mm_per_px + (W - 1) / 2, p[1] * 1000 / mm_per_px + (H - 1) / 2)
    return tid >= 0, pen_fork, pen_table, px(st["com_s"]), px(st["bbox_s"])


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def verify(out, n_gallery=30, n_iou=1000):
    import h5py
    vdir = os.path.join(out, "verify")
    os.makedirs(vdir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(out, "*-[0-9][0-9][0-9][0-9].h5")))
    if not shards:
        raise SystemExit("no shards")
    report = dict(shards=[os.path.basename(s) for s in shards])
    geo = ForkGeometry()
    # ---- 4.4a reload: shapes / dtypes / counts
    t0 = time.time()
    cols = {}
    shape_errors = []
    with h5py.File(shards[0]) as f:
        meta = json.loads(f["metadata_json"][()])
    mpp = meta["image"]["mm_per_px"]
    set_sensor(meta["config"].get("sensor", "standard"))
    _set_free_dims(meta["fields"]["free_qpos"]["shape"][0], meta["fields"]["free_qvel"]["shape"][0])
    for p in shards:
        with h5py.File(p) as f:
            n = int(f.attrs["n"])
            for k, (shp, dt) in FIELDS.items():
                dset = f[k]
                if dset.shape != (n,) + shp or dset.dtype != np.dtype(dt):
                    shape_errors.append(f"{os.path.basename(p)}:{k} {dset.shape} {dset.dtype}")
                if k not in IMAGE_FIELDS:
                    cols.setdefault(k, []).append(dset[()])
            for grp, spec in (("gel_contacts", GEL_C), ("table_contacts", TAB_C)):
                tot = int(f[grp]["count"][()].sum())
                for name, shp, dt in spec:
                    if f[grp][name].shape != (tot,) + shp:
                        shape_errors.append(f"{os.path.basename(p)}:{grp}/{name}")
            cols.setdefault("split", []).append(np.full(n, SPLITS.index(f.attrs["split"])))
            cols.setdefault("shard_path", []).append(np.full(n, p))
            cols.setdefault("index", []).append(np.arange(n))
            # per-sample sums of the gel contacts for the physics check
            gc = f["gel_contacts"]
            st, cnt, fn = gc["start"][()], gc["count"][()], gc["fn"][()]
            cols.setdefault("sum_fn", []).append(np.array([fn[a:a + c].sum() for a, c in zip(st, cnt)]))
            fx = gc["force_s"][()]
            cols.setdefault("sum_force_s", []).append(np.array([fx[a:a + c].sum(0) if c else np.zeros(3)
                                                                for a, c in zip(st, cnt)]))
            oth, part = gc["other"][()], gc["part"][()]
            cols.setdefault("parts_touched", []).append(
                [sorted(set(part[a:a + c][oth[a:a + c] == 0].tolist())) for a, c in zip(st, cnt)])
    C = {k: (np.concatenate(v) if k != "parts_touched" else sum(v, [])) for k, v in cols.items()}
    N = len(C["seed"])
    seeds_unique = len(np.unique(C["seed"])) == N
    split_sets = {s: set(C["seed"][C["split"] == i].tolist()) for i, s in enumerate(SPLITS)}
    overlap = any(split_sets[a] & split_sets[b] for a in SPLITS for b in SPLITS if a < b)
    report["reload"] = dict(samples=N, shape_dtype_errors=shape_errors, seeds_unique=bool(seeds_unique),
                            split_seed_overlap=bool(overlap),
                            per_split={s: int((C["split"] == i).sum()) for i, s in enumerate(SPLITS)},
                            read_s=time.time() - t0)
    print(f"reload: {N} samples, {len(shape_errors)} shape/dtype errors, seeds unique {seeds_unique}, "
          f"split overlap {overlap}", flush=True)

    # ---- 4.2 statistics
    stats = statistics(C, meta, vdir)
    report["statistics"] = stats

    # ---- 4.3 physics
    ph = physics_check(C)
    report["physics"] = ph

    # ---- 4.1 label check: gallery + IoU on many samples
    rng = np.random.default_rng(12345)
    pose_ids = [p["id"] for p in meta["poses"] if p["category"] in ("natural", "propped")]
    pick = []
    on = (C["sample_type"] == 0) & C["fork_visible"]
    per = int(np.ceil(n_gallery / len(pose_ids)))
    for pid in pose_ids:
        cand = np.nonzero(on & (C["pose_id"] == pid))[0]
        pick += list(rng.choice(cand, min(per, len(cand)), replace=False))
    extra = np.nonzero(C["sample_type"] != 0)[0]
    pick = pick[:n_gallery - 3] + list(rng.choice(extra, 3, replace=False))
    tiles, rows = [], []
    for g in pick:
        with h5py.File(C["shard_path"][g]) as f:
            st = load_sample(f, int(C["index"][g]))
        tile, r = label_tile(geo, st, mpp, meta)
        tiles.append(tile)
        rows.append(r)
    cols_n = 5
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols_n)
    grid = np.concatenate([np.concatenate(tiles[i:i + cols_n], 1) for i in range(0, len(tiles), cols_n)], 0)
    head = np.zeros((30, grid.shape[1], 3), np.uint8)
    cv2.putText(head, "label check: drawn FROM THE LABELS - white: fork silhouette, green: predicted contact "
                "(fork + table), magenta: COM, cyan: bbox centre, arrow: sensor origin -> COM; red thin: "
                "contact edge of the stored ground-truth depth", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(vdir, "label_gallery.png"), np.concatenate([head, grid], 0))
    # IoU over many samples with the fork visible
    cand = np.nonzero(C["fork_visible"])[0]
    sel = rng.choice(cand, min(n_iou, len(cand)), replace=False)
    ious, dd = [], []
    by_shard = {}
    for g in sel:
        by_shard.setdefault(C["shard_path"][g], []).append(int(C["index"][g]))
    for p, idx in by_shard.items():
        with h5py.File(p) as f:
            for i in sorted(idx):
                st = load_sample(f, i)
                _, pf, _, _, _ = label_maps(geo, st, mpp)
                gt = st["depth_fork"].astype(np.float32)
                ious.append(_iou(pf > 0.05, gt > 0.05))
                both = (pf > 0.1) & (gt > 0.1)
                if both.any():
                    dd.append(float(np.median(np.abs(pf[both] - gt[both]))))
    ious = np.array(ious)
    report["label_check"] = dict(
        gallery=os.path.join("verify", "label_gallery.png"), gallery_samples=rows,
        iou_label_vs_stored_fork_depth=dict(n=len(ious), mean=float(ious.mean()), p1=float(np.percentile(ious, 1)),
                                            min=float(ious.min()), below_0_9=int((ious < 0.9).sum())),
        depth_abs_diff_mm_median_of_medians=float(np.median(dd)) if dd else None)
    print(f"label check: IoU(label-predicted fork contact, stored depth_fork) over {len(ious)} samples: mean "
          f"{ious.mean():.4f}, 1st percentile {np.percentile(ious, 1):.4f}, min {ious.min():.4f}", flush=True)

    # ---- 4.4b regeneration from seeds (3 samples, this fresh process) + replay from stored state
    rng2 = np.random.default_rng(7)
    regen_rows = []
    for g in rng2.choice(N, 3, replace=False):
        r, col = regenerate(C["shard_path"][g], [int(C["index"][g])])
        r = r[0]
        r["state_replay"] = replay_state(col, C["shard_path"][g], int(C["index"][g]))
        regen_rows.append(r)
        print(f"regenerated {r['shard']}[{r['index']}] (seed {r['seed']}): identical {r['identical']} "
              f"{r['differing_fields'] or ''}; replay from stored state: {r['state_replay']}", flush=True)
    report["regeneration"] = regen_rows
    with open(os.path.join(vdir, "report.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    return report


def label_tile(geo, st, mpp, meta):
    sil, pf, pt, com_px, bbox_px = label_maps(geo, st, mpp)
    pred = np.maximum(pf, pt) > 0.05
    gt = st["depth_fork"].astype(np.float32) > 0.05
    gt_all = st["depth_gel"].astype(np.float32)
    iou = _iou(pf > 0.05, gt)

    def draw(img):
        img = img.copy()
        for mask, col, th in ((sil, (255, 255, 255), 1), (pred, (0, 255, 0), 1)):
            cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, col, th, cv2.LINE_AA)
        cs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, (0, 0, 255), 1, cv2.LINE_4)
        c0 = ((W - 1) / 2, (H - 1) / 2)
        cv2.arrowedLine(img, tuple(int(round(v)) for v in c0), tuple(int(round(v)) for v in com_px),
                        (255, 0, 255), 1, cv2.LINE_AA, tipLength=0.04)
        cv2.drawMarker(img, tuple(int(round(v)) for v in com_px), (255, 0, 255), cv2.MARKER_CROSS, 12, 2)
        cv2.drawMarker(img, tuple(int(round(v)) for v in bbox_px), (255, 255, 0), cv2.MARKER_SQUARE, 10, 1)
        return img

    rgb = draw(st["img"][..., ::-1])
    v = (np.clip(gt_all / 1.5, 0, 1) * 255).astype(np.uint8)
    dep = draw(cv2.applyColorMap(v, cv2.COLORMAP_VIRIDIS))
    pose = next(p for p in meta["poses"] if p["id"] == st["pose_id"])
    t = (f"seed {int(st['seed'])} pose {int(st['pose_id'])} {pose['category']} {SAMPLE_TYPES[st['sample_type']]}"
         f" {PARTS[st['target_part']] if st['target_part'] >= 0 else ''} {float(st['force_cmd']):.1f}N "
         f"IoU {iou:.3f}")
    bar = np.zeros((22, 2 * W, 3), np.uint8)
    cv2.putText(bar, t, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    tile = np.concatenate([bar, np.concatenate([rgb, dep], 1)], 0)
    tile = cv2.copyMakeBorder(tile, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(60, 60, 60))
    return tile, dict(seed=int(st["seed"]), pose=int(st["pose_id"]), type=SAMPLE_TYPES[st["sample_type"]],
                      iou=iou)


def statistics(C, meta, vdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = {}
    poses = {p["id"]: p for p in meta["poses"]}
    st["per_pose"] = {f"{k} ({poses[k]['category']}, {poses[k]['description']})": int((C["pose_id"] == k).sum())
                      for k in sorted(set(C["pose_id"].tolist()))}
    st["per_type"] = {t: int((C["sample_type"] == i).sum()) for i, t in enumerate(SAMPLE_TYPES)}
    on = C["sample_type"] == 0
    st["per_target_part_on_fork"] = {PARTS[i]: int((on & (C["target_part"] == i)).sum()) for i in range(len(PARTS))}
    st["per_target_part_by_pose"] = {int(k): {PARTS[i]: int((on & (C["pose_id"] == k) & (C["target_part"] == i)).sum())
                                              for i in range(len(PARTS))} for k in sorted(set(C["pose_id"].tolist()))}
    touched = {p: 0 for p in PARTS}
    for lst in C["parts_touched"]:
        for i in lst:
            touched[PARTS[i]] += 1
    st["samples_touching_part"] = touched
    st["contact_rate"] = {t: dict(fork_contact=float(C["contact_fork"][C["sample_type"] == i].mean()),
                                  table_contact=float(C["contact_table"][C["sample_type"] == i].mean()),
                                  fork_visible=float(C["fork_visible"][C["sample_type"] == i].mean()),
                                  anything_visible=float(C["contact_visible"][C["sample_type"] == i].mean()))
                          for i, t in enumerate(SAMPLE_TYPES)}
    st["flawed"] = dict(kept_with_flaw=int(C["flawed"].sum()), redrawn=int((C["retries"] > 0).sum()),
                        reasons={k: int(((C["retry_reasons"] & v) > 0).sum()) for k, v in RETRY.items()},
                        saturated_3mm=int(C["saturated"].sum()), static_unsettled=int((~C["settled"]).sum()))
    fb = [0.2, 0.5, 1, 2, 5, 10.01]
    pressed = C["sample_type"] != 2
    st["would_move"] = {}
    for k in sorted(set(C["pose_id"].tolist())):
        sel = C["pose_id"] == k
        row = dict(all=float(C["would_move"][sel].mean()), n=int(sel.sum()),
                   hover=float(C["would_move"][sel & ~pressed].mean()) if (sel & ~pressed).any() else None)
        for a, b in zip(fb[:-1], fb[1:]):
            s2 = sel & pressed & (C["force_cmd"] >= a) & (C["force_cmd"] < b) & C["contact_fork"]
            row[f"fork_contact_{a}-{b:g}N"] = (float(C["would_move"][s2].mean()), int(s2.sum())) if s2.any() else None
        row["no_fork_contact_presses"] = (float(C["would_move"][sel & pressed & ~C["contact_fork"]].mean())
                                          if (sel & pressed & ~C["contact_fork"]).any() else None)
        st["would_move"][int(k)] = row
    dist_com = np.linalg.norm(C["com_s"][:, :2], axis=1) * 1000
    dist_bb = np.linalg.norm(C["bbox_s"][:, :2], axis=1) * 1000
    q = lambda x: {f"p{p}": float(np.percentile(x, p)) for p in (0, 5, 25, 50, 75, 95, 100)}
    st["distance_to_centre_mm"] = dict(com_xy=q(dist_com), bbox_xy=q(dist_bb),
                                       com_xy_on_fork=q(dist_com[on]), com_xy_table=q(dist_com[C["sample_type"] == 1]))
    st["force"] = dict(cmd=q(C["force_cmd"][pressed]), touch=q(C["force_touch"][pressed]))
    st["n_tines_on_pad"] = {int(k): int((C["n_tines"] == k).sum()) for k in range(5)}
    st["tilt_deg"] = q(C["tilt_deg"])
    # figure
    fig, ax = plt.subplots(3, 4, figsize=(20, 13))
    ax = ax.ravel()
    ax[0].hist2d(C["fork_place"][:, 0] * 1000, C["fork_place"][:, 1] * 1000, bins=30)
    ax[0].set_title("fork bbox centre, world xy [mm]")
    ax[1].hist2d(C["sensor_pos_w"][:, 0] * 1000, C["sensor_pos_w"][:, 1] * 1000, bins=40)
    ax[1].set_title("sensor origin, world xy [mm]")
    ax[2].hist(np.degrees(C["fork_place"][:, 2]), 36, alpha=0.6, label="fork yaw")
    ax[2].hist(np.degrees(C["sensor_rpy_w"][:, 2]), 36, alpha=0.6, label="sensor yaw")
    ax[2].legend(), ax[2].set_title("yaw [deg]")
    ax[3].hist(C["force_cmd"][pressed], np.geomspace(0.2, 10, 30), alpha=0.6, label="commanded")
    ax[3].hist(C["force_touch"][pressed], np.geomspace(0.2, 10, 30), alpha=0.6, label="touch sensor")
    ax[3].set_xscale("log"), ax[3].legend(), ax[3].set_title("force [N] (presses)")
    for i, t in enumerate(SAMPLE_TYPES):
        ax[4].hist(dist_com[C["sample_type"] == i], 50, alpha=0.6, label=t)
    ax[4].legend(), ax[4].set_title("distance sensor origin -> COM, sensor xy [mm]")
    ax[5].hist2d(C["com_s"][on, 0] * 1000, C["com_s"][on, 1] * 1000, bins=50, range=[[-90, 90], [-90, 90]])
    ax[5].set_title("COM in the sensor frame, on_fork [mm]")
    ax[6].hist(C["tilt_deg"], 30), ax[6].set_title("sensor tilt [deg]")
    ax[7].hist(C["peak_fork_mm"][on], 40), ax[7].set_title("peak fork indentation, on_fork [mm]")
    ks = sorted(st["would_move"])
    centers = np.sqrt(np.array(fb[:-1]) * np.array(fb[1:]))
    for k in ks:
        ys = [st["would_move"][k][f"fork_contact_{a}-{b:g}N"] for a, b in zip(fb[:-1], fb[1:])]
        ax[8].plot(centers, [y[0] if y else np.nan for y in ys], "o-", label=f"pose {k}")
    ax[8].set_xscale("log"), ax[8].legend(), ax[8].set_title("would_move rate vs force (fork contact)")
    labels = list(st["per_pose"].keys())
    ax[9].bar(range(len(labels)), list(st["per_pose"].values()))
    ax[9].set_xticks(range(len(labels)), [l.split(" (")[0] + " " + l.split("(")[1].split(",")[0] for l in labels])
    ax[9].set_title("samples per pose")
    ax[10].bar(PARTS, [st["per_target_part_on_fork"][p] for p in PARTS], alpha=0.6, label="target (on_fork)")
    ax[10].bar(PARTS, [st["samples_touching_part"][p] for p in PARTS], alpha=0.6, label="touched (any type)")
    ax[10].legend(), ax[10].set_title("fork parts")
    ax[11].bar(SAMPLE_TYPES, [st["per_type"][t] for t in SAMPLE_TYPES]), ax[11].set_title("sample types")
    fig.tight_layout()
    fig.savefig(os.path.join(vdir, "statistics.png"), dpi=80)
    plt.close(fig)
    print("statistics:", json.dumps({k: st[k] for k in ("per_pose", "per_type", "per_target_part_on_fork",
                                                        "contact_rate", "flawed")}, indent=1), flush=True)
    return st


def physics_check(C):
    pressed = C["sample_type"] != 2
    err = np.abs(C["sum_fn"] - C["force_touch"])
    rel = err / np.maximum(C["force_touch"], 1e-9)
    big = C["force_touch"] > 0.05
    tc = pressed & C["settled"] & ~C["flawed"]
    cmd_err = np.abs(C["force_touch"][tc] / C["force_cmd"][tc] - 1)
    # the press actuator pushes along world -z; the contacts push the gel back: the world-vertical
    # component of the total force on the gel must equal the command (the touch sensor sums the
    # NORMAL forces, which exceed it when the gel's rim pushes on a fork side)
    fz = np.array([(quat2mat(q) @ f)[2] for q, f in zip(C["sensor_quat_w"], C["sum_force_s"])])
    vz_err = np.abs(fz[tc] / C["force_cmd"][tc] - 1)
    still = pressed & ~C["would_move"] & C["free_settled"]
    ph = dict(sum_contact_normal_vs_touch=dict(n=int(len(err)), max_abs_n=float(err.max()),
                                               max_rel_where_touch_gt_0_05N=float(rel[big].max()) if big.any() else None,
                                               median_rel=float(np.median(rel[big])) if big.any() else None),
              touch_vs_command_settled=dict(n=int(tc.sum()), median_rel=float(np.median(cmd_err)),
                                            p99_rel=float(np.percentile(cmd_err, 99)), max_rel=float(cmd_err.max())),
              vertical_gel_force_vs_command_settled=dict(n=int(tc.sum()), median_rel=float(np.median(vz_err)),
                                                         p99_rel=float(np.percentile(vz_err, 99)),
                                                         max_rel=float(vz_err.max())),
              free_fork_force_balance_residual_n=dict(
                  n=int(still.sum()), median=float(np.median(C["free_force_residual"][still])),
                  p99=float(np.percentile(C["free_force_residual"][still], 99)),
                  max=float(C["free_force_residual"][still].max()),
                  p99_relative_to_touch=float(np.percentile(C["free_force_residual"][still] /
                                                            np.maximum(C["free_force_touch"][still], 0.05), 99)))
              if still.any() else dict(n=0, note="no free-fork run in this dataset (static fork, collected without --free-run)"),
              hover_touch_max_n=float(C["force_touch"][~pressed].max()) if (~pressed).any() else None)
    print("physics:", json.dumps(ph, indent=1), flush=True)
    return ph


# ================================================================ CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("poses")
    a.add_argument("--out", default=OUT)
    a.add_argument("--drops", type=int, default=500)
    a.add_argument("--workers", type=int, default=24)
    a = sub.add_parser("collect")
    a.add_argument("--out", default=OUT)
    a.add_argument("--n", type=int, default=10_000)
    a.add_argument("--splits", type=float, nargs=3, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    a.add_argument("--only-split", default=None, choices=SPLITS,
                   help="put all --n samples into this split (e.g. train2 / test2: new seed ranges)")
    a.add_argument("--shard-size", type=int, default=1000)
    a.add_argument("--workers", type=int, default=16)
    a.add_argument("--fracs", type=float, nargs=3, default=[0.7, 0.2, 0.1], metavar=("ON_FORK", "TABLE", "HOVER"))
    a.add_argument("--include-edge-poses", action="store_true",
                   help="also sample the propped side-edge poses (fork held on its edge by a fixture)")
    a.add_argument("--propped-frac", type=float, default=0.1, help="share of propped samples with --include-edge-poses")
    a.add_argument("--free-run", action="store_true",
                   help="also run each press with a free fork (would_move); off: the fork is clamped to the table")
    a.add_argument("--part-weights", type=float, nargs=5, default=[1, 1, 1, 1, 1], metavar=tuple(p.upper() for p in PARTS))
    a.add_argument("--fork-xy-range-mm", type=float, default=30.0)
    a.add_argument("--tilt-max-deg", type=float, default=3.0)
    a.add_argument("--force-range", type=float, nargs=2, default=[0.2, 10.0])
    a.add_argument("--sensor", default="standard", choices=["standard", "mini"],
                   help="sensor size: standard 32 x 24 mm (320 x 240 px), mini = GelSight Mini 18.6 x 14.4 mm (186 x 144 px)")
    a = sub.add_parser("verify")
    a.add_argument("--out", default=OUT)
    a = sub.add_parser("regen")
    a.add_argument("shard")
    a.add_argument("index", type=int, nargs="+")
    args = ap.parse_args()
    if args.cmd == "poses":
        stable_poses(os.path.join(args.out, "poses"), args.drops, args.workers)
    elif args.cmd == "collect":
        cfg = Config(frac_on_fork=args.fracs[0], frac_table=args.fracs[1], frac_hover=args.fracs[2],
                     propped_frac=args.propped_frac if args.include_edge_poses else 0.0, free_run=args.free_run,
                     part_weights=dict(zip(PARTS, args.part_weights)),
                     fork_xy_range_m=args.fork_xy_range_mm / 1000, tilt_max_deg=args.tilt_max_deg,
                     force_min_n=args.force_range[0], force_max_n=args.force_range[1], sensor=args.sensor)
        set_sensor(args.sensor)
        os.makedirs(args.out, exist_ok=True)
        collect(args.out, args.n, args.splits, args.shard_size, args.workers, cfg, args.only_split)
    elif args.cmd == "verify":
        verify(args.out)
    elif args.cmd == "regen":
        res, _ = regenerate(args.shard, args.index)
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
