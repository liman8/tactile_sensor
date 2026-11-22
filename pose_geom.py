"""
CAD geometry for the full-pose search (pose addendum C3 / C4 / C6): every object x natural stable pose, in the frame of
the pose convention (objects.json "conventions"): origin = the centre of mass, +x = the long axis towards the head.

    import pose_geom as pg
    G = pg.ObjectPose("knife_stl", 0)            # object, stable-pose id (natural)
    G.footprint_hits(k)                          # (221, 221) bool: CoM offsets (sensor frame, 1 mm) whose object touches the pad, yaw bin k
    G.labels                                     # (81, 201) int: region label of the canonical top view (information gain)
    G.height                                     # (401, 1001) float: top-surface height above the table, canonical frame, 0.2 mm
    G.points(R, t)                               # CAD points (ADD) at a pose

Frames.
  canonical: the object in its stable pose, turned about the vertical so that +x points along the table x, CoM at the
             origin in x-y, z = height above the table.
  sensor (level sensor of yaw gamma): x = image column, y = image row. An object whose +x makes the sensor-frame angle psi
             (the yaw head's angle) has its canonical point p at the sensor offset M(psi) p with
             M(psi) = [[cos psi, sin psi], [sin psi, -cos psi]] (a reflection: the sensor frame is left-handed in x-y seen
             from above); M(psi) is its own inverse. World yaw of +x: psi_w = gamma - psi.
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np
import trimesh

import objects

HERE = os.path.dirname(os.path.abspath(__file__))
YAW_BINS = 72
YAW_BIN = 2 * np.pi / YAW_BINS
PAD_HALF_MM = (9.3, 7.2)                         # Mini pad at the network input (93 x 72 px of 0.2 mm), sensor x / y
HIT_HALF = 110                                   # footprint_hits window: offsets -110..110 mm
LAB_X, LAB_Y = 100, 40                           # label / nearest-label window (1 mm): x -100..100, y -40..40
H_RES = 0.2                                      # height map resolution (mm), window x -100..100, y -40..40
EMPTY = -1


def M(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, s], [s, -c]])


def bin_centre(k):
    return (np.asarray(k) + 0.5) * YAW_BIN


def rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def quat2mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


_MESH = {}


def mesh(name):
    if name not in _MESH:
        _MESH[name] = trimesh.load(os.path.join(objects.mesh_dir(name), f"{name}_visual.stl"), process=True)
    return _MESH[name]


def stable_poses(name):
    return {p["id"]: p for p in json.load(open(os.path.join(HERE, objects.get(name)["poses"]["mini"])))["poses"]}


def _fill(mask, tris_px):
    for t in tris_px:
        cv2.fillPoly(mask, [np.round(t * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return mask


class ObjectPose:
    def __init__(self, name, pose_id, add_points=2000):
        self.name, self.pose_id = name, int(pose_id)
        o = objects.get(name)
        p = stable_poses(name)[pose_id]
        self.R_rest = quat2mat(p["quat_wxyz"])
        self.rest_pos = np.array(p["rest_pos_m"]) * 1000.0
        self.psi_rest = float(np.arctan2(self.R_rest[1, 0], self.R_rest[0, 0]))
        self.R_c = rz(-self.psi_rest) @ self.R_rest                         # canonical orientation of the body
        self.com_b = np.array(o["com_mm"])
        self.edge = abs(self.R_rest[2, 2]) < 0.3                           # standing on a side edge (STL z horizontal)
        m = mesh(name)
        V = (m.vertices - self.com_b) @ self.R_c.T
        self.z_com = float(-((m.vertices - self.com_b) @ self.R_rest.T)[:, 2].min())   # CoM height above the table
        V[:, 2] += self.z_com                                              # z = height above the table
        self.V, self.F = V, m.faces
        pts = np.vstack([trimesh.sample.sample_surface(m, add_points, seed=0)[0], m.vertices])
        self.add_pts = (pts - self.com_b) @ self.R_c.T + [0, 0, self.z_com]   # canonical, CoM at x = y = 0
        self._labels()
        self._height()
        self._hits = {}

    # ---- regions for the information gain: what a press there would look like
    def _labels(self):
        name = self.name
        sym = objects.get(name)["symmetric"]
        W, H = 2 * LAB_X + 1, 2 * LAB_Y + 1
        fp = _fill(np.zeros((H, W), np.uint8), self.V[self.F][:, :, :2] + [LAB_X, LAB_Y])
        self.footprint = fp.astype(bool)
        ys, xs = np.mgrid[-LAB_Y:LAB_Y + 1, -LAB_X:LAB_X + 1]
        x_stl = xs + self.com_b[0]                                         # canonical x = STL x - com x (no turn in x)
        shared = name in ("fork2_stl", "knife_stl", "spoon_stl")
        # geometry group: the flat handle / neck of Fork, Knife, Spoon is the same shape; the head of each object and
        # fork_stl (bent) are distinct. Edge poses of an asymmetric object differ by which edge is up.
        grp = np.where((x_stl < 1.0) & shared, 9, objects.object_id(name))
        if self.edge and not sym:
            grp = grp * 10 + self.pose_id
        xb = np.floor(x_stl / 5.0).astype(int) + 40
        yb = np.zeros_like(xb) if self.edge else np.clip(np.floor(ys / 2.0).astype(int), -6, 6) + 6
        lab = ((grp * 2 + int(self.edge)) * 100 + xb) * 20 + yb
        from scipy.ndimage import distance_transform_edt
        _, (iy, ix) = distance_transform_edt(~self.footprint, return_indices=True)
        self.labels = lab[iy, ix]                                          # the nearest footprint pixel's label

    # ---- top-surface height (above the table) in the canonical frame, for the expected depth map (C4)
    def _height(self):
        import collect_dataset as cd
        W, H = int(2 * 100 / H_RES) + 1, int(2 * 40 / H_RES) + 1
        P = np.c_[(self.V[:, 0] + 100) / H_RES, (self.V[:, 1] + 40) / H_RES, self.V[:, 2]]
        z, tid = cd.rasterize(P, self.F, H, W, keep="max")
        self.height = np.where(np.isfinite(z), z, 0.0).astype(np.float32)

    def height_at(self, p):
        """Canonical (x, y) mm (..., 2) -> top-surface height (mm), bilinear, 0 off the object."""
        from scipy.ndimage import map_coordinates
        return map_coordinates(self.height, [(p[..., 1] + 40) / H_RES, (p[..., 0] + 100) / H_RES], order=1, cval=0.0)

    # ---- which CoM offsets make the object touch the pad, per yaw bin (sensor frame, 1 mm)
    def footprint_hits(self, k):
        if k not in self._hits:
            psi = bin_centre(k)
            S = 2 * HIT_HALF + 1
            Pxy = self.V[:, :2] @ M(psi).T + HIT_HALF
            m = _fill(np.zeros((S, S), np.uint8), Pxy[self.F])
            ker = np.ones((2 * int(round(PAD_HALF_MM[1])) + 1, 2 * int(round(PAD_HALF_MM[0])) + 1), np.uint8)
            self._hits[k] = cv2.dilate(m, ker).astype(bool)               # pad centre offset d = q - c with a hit
        return self._hits[k]

    def height_hits(self, k):
        """(221, 221) float32: the highest top-surface point (mm above the table) of the object under a pad centred at
        CoM offset d = q - c (sensor frame, 1 mm), yaw bin k; 0 where the pad misses the object (the table). The CAD's
        top height is sampled every 0.5 mm (max of 4 sub-samples per 1 mm cell) and max-filtered over the pad."""
        if not hasattr(self, "_hh"):
            self._hh = {}
            f = os.path.join(HERE, "results_multi", "pose_cache", f"height_hits_{self.name}_{self.pose_id}.npy")
            if os.path.exists(f):
                arr = np.load(f)
                self._hh = {kk: arr[kk] for kk in range(YAW_BINS)}
        if k not in self._hh:
            psi = bin_centre(k)
            g = np.arange(2 * HIT_HALF + 1) - HIT_HALF
            gx, gy = np.meshgrid(g, g)
            z = np.zeros(gx.shape, np.float32)
            for ox, oy in ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)):
                u = np.stack([gx + ox, gy + oy], -1)
                z = np.maximum(z, self.height_at(u @ M(psi).T).astype(np.float32))
            ker = np.ones((2 * int(round(PAD_HALF_MM[1])) + 1, 2 * int(round(PAD_HALF_MM[0])) + 1), np.uint8)
            self._hh[k] = cv2.dilate(z, ker)
        return self._hh[k]

    def cache_height_hits(self):
        f = os.path.join(HERE, "results_multi", "pose_cache", f"height_hits_{self.name}_{self.pose_id}.npy")
        if not os.path.exists(f):
            os.makedirs(os.path.dirname(f), exist_ok=True)
            np.save(f, np.stack([self.height_hits(k) for k in range(YAW_BINS)]).astype(np.float32))

    def points(self, c_xy, psi_w, gamma=None):
        """CAD points (mm) in the table frame: canonical points turned to world yaw psi_w, CoM at c_xy."""
        P = self.add_pts @ rz(psi_w).T
        P[:, :2] += np.asarray(c_xy)
        return P


_CACHE = {}


def get(name, pose_id):
    k = (name, int(pose_id))
    if k not in _CACHE:
        _CACHE[k] = ObjectPose(name, pose_id)
    return _CACHE[k]


def add(G_true, c_true, psi_true, G_est, c_est, psi_est):
    """ADD (mm): mean distance between corresponding CAD points at the true and the estimated pose (same object and pose:
    corresponding points; otherwise the estimated object's points are compared by nearest neighbour, i.e. ADD-S)."""
    A = G_true.points(c_true, psi_true)
    B = G_est.points(c_est, psi_est)
    if G_true.name == G_est.name and G_true.pose_id == G_est.pose_id:
        return float(np.linalg.norm(A - B, axis=1).mean())
    return add_s(G_true, c_true, psi_true, G_est, c_est, psi_est)


def add_s(G_true, c_true, psi_true, G_est, c_est, psi_est):
    """ADD-S (mm): mean over the true pose's points of the distance to the nearest point of the estimated pose."""
    from scipy.spatial import cKDTree
    A = G_true.points(c_true, psi_true)
    B = G_est.points(c_est, psi_est)
    return float(cKDTree(B).query(A)[0].mean())
