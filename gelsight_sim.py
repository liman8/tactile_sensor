"""
GelSight-like tactile sensor simulation on top of MuJoCo (>= 4.0).

Two setups with the same sensor, optics and tactile images:

    mode="moving_sensor"  gelsight_moving.xml  (DEFAULT) the object lies still on the table;
                                               the sensor, gel-down under a moving "arm",
                                               presses onto it. Starts with fork_stl
                                               (ForkMilimeter.stl) on the table.
    mode="fixed_sensor"   gelsight_fork.xml    the sensor sits on the table gel-up; the
                                               object hangs from a moving "indenter" and
                                               is pressed onto it (the original setup;
                                               starts with fork_dinner, has no fork_stl)

Pipeline (what a real GelSight does, approximated):

    MuJoCo contact physics            object sinks into the compliant gel pad
            |
    orthographic depth camera         looks through the gel from inside the sensor; only
    (rendered from inside the sensor) the active object is drawn  ->  object surface height
            |
    gel deformation model             indentation = how far the object is past the
                                      undeformed gel surface, plus an elastic "skirt"
                                      around the contact (Gaussian spread)
            |
    photometric shading               3 coloured LEDs at 120 deg (R, G, B), Lambertian
                                      reflective skin -> RGB tactile image, plus black
                                      markers that move with the gel, optics blur, noise

Typical use:

    sim = GelSightSim()                                   # moving sensor, fork_stl on the table
    info = sim.press_relative(3.0, yaw=0.3)               # sensor on the fork's tines (fork_stl_face)
    rgb = sim.tactile_image()            # what the sensor camera sees (H x W x 3 uint8)
    ref = sim.reference_image()          # no-contact frame, used for differencing
    depth_mm = sim.indentation_mm()      # ground-truth gel indentation (mm)
    sim.place_object("fork_stl", x=0.01, y=-0.005, yaw=0.3)      # lay it elsewhere on the table
    info = sim.press(force=4.0, x=0.012, y=-0.004, yaw=0.9)      # moves the SENSOR (world frame)
    info = sim.press_relative(4.0, x=0.002, yaw=0.6)  # object pose in the sensor frame
    sim.place_object("spoon")                         # any other object

    sim = GelSightSim(mode="fixed_sensor")                # the original fixed sensor
    sim.set_tool("fork_dinner")
    info = sim.press(force=4.0, x=0.002, yaw=0.6, pitch=0.05)    # moves the TOOL
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

# Offscreen tactile renderer: EGL on Linux unless the user chose a GL backend already. This
# only works if it runs before the first `import mujoco` (scripts set it themselves too);
# the interactive viewer window always uses GLFW.
if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import cv2
import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.join(HERE, "gelsight_fork.xml")      # fixed sensor, moving tool
MOVING_XML = os.path.join(HERE, "gelsight_moving.xml")     # moving sensor, object on the table
MODES = ("fixed_sensor", "moving_sensor")

TOOLS = [
    "fork_dinner", "fork_dessert", "fork_salad", "fork_wide", "fork_3wide", "fork_plastic",
    "spoon", "knife_flat", "knife_edge", "pen", "comb", "ball", "coin", "cube",
    "cal_ball",
]
# objects built from STL files by stl_to_mjcf.py (only in gelsight_moving.xml): a visual-only
# mesh plus convex collision parts. fork_stl is a training fork of the moving-sensor detector
# (train_detector.TRAIN_FORKS_MOVING). Other bodies that hang off the worldbody and have a
# "<name>_face" site are picked up as objects too.
MESH_TOOLS = ["fork_stl"]
DEFAULT_OBJECT = "fork_stl"    # object a new GelSightSim starts with (falls back to the XML's active one)
DEFAULT_MODE = "moving_sensor"  # fork_stl exists only there


def _extra_objects(names_with_face):
    """Objects beyond TOOLS + MESH_TOOLS: (body name, has a "<name>_face" site) pairs."""
    return [n for n, face in names_with_face if face and n not in TOOLS + MESH_TOOLS]
FORKS = {"fork_dinner", "fork_dessert", "fork_salad", "fork_wide", "fork_3wide", "fork_plastic", "fork_stl"}
JOINTS = ["slide_x", "slide_y", "slide_z", "yaw", "pitch", "roll"]

# Sensor sizes (moving_sensor only for anything but "standard"). Both at 0.1 mm per pixel.
#   standard: the sensor of gelsight_fork.xml / gelsight_moving.xml, 32 x 24 mm gel and field of view.
#   mini:     the GelSight Mini's field of view (datasheet 18.6 x 14.3 mm): 186 x 144 px = 18.6 x 14.4 mm.
#             Gel pad, acrylic, touch site and housing shrink around it (same 3 mm walls); the sensor
#             keeps the standard sensor's mass and inertia, so the gel's force -> indentation law is the
#             same (MuJoCo scales soft contacts by body_invweight0). Gel thickness stays 3 mm: in this
#             model the gel's compliance comes from solref/solimp and the moving mass, not its thickness.
SENSORS = {"standard": dict(width=320, height=240), "mini": dict(width=186, height=144)}
SENSOR_ENV = "GELSIGHT_SENSOR"      # default sensor for GelSightSim(sensor=None) (inherited by worker processes)
SENSOR_WALL_M = 0.003


def default_sensor() -> str:
    s = os.environ.get(SENSOR_ENV, "standard")
    if s not in SENSORS:
        raise ValueError(f"{SENSOR_ENV}={s!r}: choose from {list(SENSORS)}")
    return s


def _resize_sensor(spec, hx, hy, inertial):
    """Gel of 2hx x 2hy [m] in body "gelsight": gel, acrylic, touch site, housing, LEDs (visual) and the
    orthographic camera (fovy = full vertical extent). inertial = (mass, ipos, iquat, inertia) to keep."""
    wt = SENSOR_WALL_M
    for name in ("gel", "acrylic"):
        g = spec.geom(name)
        g.size = [hx, hy, g.size[2]]
    st = spec.site("gel_touch")
    st.size = [hx + 1e-4, hy + 1e-4, st.size[2]]
    base = spec.geom("base")
    base.size = [hx + 2 * wt, hy + 2 * wt, base.size[2]]
    for name, sx in (("wall_left", -1), ("wall_right", 1)):
        g = spec.geom(name)
        g.pos = [sx * (hx + wt), 0.0, g.pos[2]]
        g.size = [wt, hy + 2 * wt, g.size[2]]
    for name, sy in (("wall_back", 1), ("wall_front", -1)):
        g = spec.geom(name)
        g.pos = [0.0, sy * (hy + wt), g.pos[2]]
        g.size = [hx, wt, g.size[2]]
    k = hy / 0.012
    for name in ("led_r", "led_g", "led_b"):
        g = spec.geom(name)
        g.pos = [g.pos[0] * k, g.pos[1] * k, g.pos[2]]
    b = spec.geom("cam_board")
    b.size = [min(b.size[0], hx - 1e-3), min(b.size[1], hy - 1e-3), b.size[2]]
    spec.camera("gelsight_cam").fovy = 2 * hy
    body = spec.body("gelsight")
    body.explicitinertial = True
    body.mass, body.ipos, body.iquat, body.inertia = inertial


def sensor_spec_edit(sensor: str, xml_path: str = None):
    """function(MjSpec) that turns the standard sensor into `sensor` (None for "standard")."""
    if sensor == "standard":
        return None
    prof = SENSORS[sensor]
    m0 = mujoco.MjModel.from_xml_path(xml_path or MOVING_XML)
    b = m0.body("gelsight").id
    inertial = (float(m0.body_mass[b]), m0.body_ipos[b].copy(), m0.body_iquat[b].copy(), m0.body_inertia[b].copy())
    hx, hy = prof["width"] * 5e-5, prof["height"] * 5e-5          # half sizes [m] at 0.1 mm/px
    return lambda spec: _resize_sensor(spec, hx, hy, inertial)


def sensor_model(sensor: str, xml_path: str = None):
    """Compiled model of the moving-sensor scene with `sensor` (provenance hashes)."""
    xml_path = xml_path or MOVING_XML
    if sensor == "standard":
        return mujoco.MjModel.from_xml_path(xml_path)
    spec = mujoco.MjSpec.from_file(xml_path)
    sensor_spec_edit(sensor, xml_path)(spec)
    return spec.compile()


ACTIVE_GROUP, PARKED_GROUP = 1, 5
TABLE_GROUP = 0                # the table (moving_sensor); drawn by the tactile camera only with render_table=True
COLLISION_GROUP = 3            # convex parts of mesh objects: collide when active, never rendered
FREE_INVWEIGHT_FRAC = 0.05     # free_objects: active object's invweight0, relative to the sensor's
FREE_CREEP_TOL = 1e-3          # free_objects: object speed [m/s] still counted as settled


@dataclass
class OpticsConfig:
    """Parameters of the synthetic tactile image."""
    height: int = 240                      # px  (camera fovy = 24 mm -> 0.1 mm/px)
    width: int = 320                       # px
    max_indent_mm: float = 3.0             # clip (gel is 3 mm thick)
    spread_sigma_mm: float = 0.45          # elastic skirt of the gel around a contact
    spread_gain: float = 0.9
    edge_sigma_mm: float = 0.08            # rounds off perfectly sharp CAD edges
    led_azimuth_deg: tuple = (90.0, 210.0, 330.0)   # R, G, B (image frame)
    led_elevation_deg: float = 35.0
    background_rgb: tuple = (118.0, 124.0, 136.0)
    led_gain: tuple = (150.0, 150.0, 165.0)
    led_falloff: float = 0.12              # brighter near each LED (spatial non-uniformity)
    optics_blur_px: float = 0.6
    noise_std: float = 1.5                 # grey levels
    markers: bool = True
    marker_pitch_mm: float = 2.0
    marker_radius_mm: float = 0.28
    marker_darkness: float = 0.85
    marker_shift_gain_mm: float = 0.35     # lateral marker motion per unit surface slope


@dataclass
class PressInfo:
    tool: str
    force_cmd: float
    gel_force: float          # touch sensor reading [N]
    max_indent_mm: float      # ground-truth peak indentation
    contact_height_m: float   # slide_z at first contact
    settled: bool
    steps: int
    visible_tines: int = 0    # forks only: tines touching the gel inside the camera FOV
    pose: dict = field(default_factory=dict)          # joint targets: tool (fixed) / sensor (moving)
    object_pose: dict = field(default_factory=dict)   # moving_sensor: where the object lies
    table_contact: bool = False     # the sensor (gel or housing) touches the table
    housing_contact: bool = False   # a sensor part other than the gel touches something
    object_shift_mm: float = 0.0    # free_objects: how far the object's centre of mass moved


def _quat_to_mat(q) -> np.ndarray:
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


class GelSightSim:
    def __init__(self, xml_path: str | None = None, optics: OpticsConfig | None = None,
                 seed: int | None = None, mode: str | None = None, free_objects: bool = False,
                 spec_edit=None, render_table: bool = False, sensor: str | None = None):
        """mode: "moving_sensor" (default) or "fixed_sensor". With an explicit xml_path the
        mode is read from the model (an "arm" body means moving_sensor).
        Starts with DEFAULT_OBJECT (fork_stl) active if the model has it, else with the XML's
        active object (fork_dinner in gelsight_fork.xml).
        free_objects (moving_sensor only): every object gets a freejoint, so it can slide
        or tip under the sensor instead of being fixed to the table.
        spec_edit: optional function(MjSpec) that edits the model before it is compiled
        (experiments, e.g. verify_stl.py's SDF variant of fork_stl).
        render_table (moving_sensor only): the tactile camera also draws the table, so a gel
        pressed onto the table (alone, or next to the object with a tilted sensor) shows
        that contact; by default it draws only the active object (collect_dataset.py uses True).
        sensor: "standard" (32 x 24 mm) or "mini" (GelSight Mini field of view, 18.6 x 14.4 mm), see
        SENSORS; None = the GELSIGHT_SENSOR environment variable, else "standard"."""
        if mode is not None and mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode is None and (xml_path is None or free_objects):
            mode = DEFAULT_MODE if xml_path is None else "moving_sensor"
        if free_objects and mode != "moving_sensor":
            raise ValueError("free_objects=True needs mode='moving_sensor'")
        if xml_path is None:
            xml_path = MOVING_XML if mode == "moving_sensor" else DEFAULT_XML
        sensor = default_sensor() if sensor is None else sensor
        if sensor not in SENSORS:
            raise ValueError(f"sensor must be one of {list(SENSORS)}, got {sensor!r}")
        self.sensor = sensor
        if sensor != "standard":
            if mode != "moving_sensor":
                raise ValueError(f"sensor={sensor!r} needs mode='moving_sensor'")
            prof = SENSORS[sensor]
            if optics is not None and (optics.height, optics.width) != (prof["height"], prof["width"]):
                raise ValueError(f"sensor {sensor!r} renders {prof['width']} x {prof['height']} px")
            import dataclasses
            optics = dataclasses.replace(optics or OpticsConfig(), height=prof["height"], width=prof["width"])
            edits = [e for e in (sensor_spec_edit(sensor, xml_path), spec_edit) if e is not None]
            spec_edit = lambda spec: [e(spec) for e in edits]
        if free_objects or spec_edit is not None:
            spec = mujoco.MjSpec.from_file(xml_path)
            sites = {st.name for st in spec.sites}
            extra = _extra_objects([(b.name, f"{b.name}_face" in sites) for b in spec.worldbody.bodies])
            for t in TOOLS + MESH_TOOLS + extra if free_objects else []:
                if spec.body(t) is not None:
                    spec.body(t).add_freejoint().name = f"{t}_free"
            if spec_edit is not None:
                spec_edit(spec)
            self.model = spec.compile()
        else:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        m = self.model
        found = "moving_sensor" if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "arm") >= 0 else "fixed_sensor"
        if mode is not None and mode != found:
            raise ValueError(f"{xml_path} is a {found} model, not {mode}")
        self.mode = found
        self.moving = found == "moving_sensor"
        self.free_objects = free_objects
        self.data = mujoco.MjData(m)
        self.opt = optics or OpticsConfig()
        self.rng = np.random.default_rng(seed)

        self.cam_id = m.camera("gelsight_cam").id
        self.gel_id = m.geom("gel").id
        self.table_id = m.geom("table").id
        self.sensor_body = m.body("gelsight").id
        self.sensor_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.sensor_body}
        self.jnt = {n: m.joint(n).qposadr[0] for n in JOINTS}
        self.dof_z = m.joint("slide_z").dofadr[0]
        self.act = {n: m.actuator(n).id for n in ["x", "y", "press", "yaw", "pitch", "roll"]}
        self.touch_adr = m.sensor("gel_force").adr[0]
        self.z_range = m.jnt_range[m.joint("slide_z").id].copy()
        # press() starts this high (clear of every object); lift()/set_pose() default height
        self.z_start = 0.06 if self.moving else 0.02
        self.z_home = float(m.key_qpos[0, self.jnt["slide_z"]])

        # objects in this model (mesh objects exist only in gelsight_moving.xml) -> geom ids
        top = [m.body(b).name for b in range(1, m.nbody) if m.body_parentid[b] == 0]
        extra = _extra_objects([(n, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{n}_face") >= 0) for n in top])
        self.objects = [t for t in TOOLS + MESH_TOOLS + extra if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, t) >= 0]
        self.tool_geoms = {}
        self.tool_body = {}
        # mesh objects: geoms in COLLISION_GROUP only collide, the body's other geoms only render
        self.collision_geoms, self.visual_geoms = set(), set()
        for t in self.objects:
            bid = m.body(t).id
            self.tool_body[t] = bid
            self.tool_geoms[t] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
            col = {g for g in self.tool_geoms[t] if m.geom_group[g] == COLLISION_GROUP}
            if col:
                self.collision_geoms |= col
                self.visual_geoms |= set(self.tool_geoms[t]) - col
        self.tool = None

        if self.moving:
            # resting pose of every object at the table origin (from the XML) and where it lies now
            self.rest_pos, self.rest_quat, self.object_top = {}, {}, {}
            self.face_site = {t: m.site(f"{t}_face").id for t in self.objects}
            self.object_pose = {t: dict(x=0.0, y=0.0, yaw=0.0) for t in self.objects}
            self._obj_qadr, self._obj_dadr = {}, {}
            for t, b in self.tool_body.items():
                if free_objects:
                    j = m.body_jntadr[b]
                    self._obj_qadr[t], self._obj_dadr[t] = m.jnt_qposadr[j], m.jnt_dofadr[j]
                    a = self._obj_qadr[t]
                    self.rest_pos[t], self.rest_quat[t] = m.qpos0[a:a + 3].copy(), m.qpos0[a + 3:a + 7].copy()
                else:
                    self.rest_pos[t], self.rest_quat[t] = m.body_pos[b].copy(), m.body_quat[b].copy()
            self._invweight0 = m.body_invweight0.copy()

        mujoco.mj_resetDataKeyframe(m, self.data, 0)
        mujoco.mj_forward(m, self.data)
        # camera -> undeformed gel surface, along the viewing axis, from the SENSOR frame: the
        # camera and the gel are both fixed in body "gelsight", so this is a sensor constant
        # (it does not change when the arm moves the sensor)
        self.d0 = self._camera_to_gel()
        self.fov_h_mm = float(m.cam_fovy[self.cam_id]) * 1000.0   # orthographic: full extent
        self.mm_per_px = self.fov_h_mm / self.opt.height
        self.fov_w_mm = self.mm_per_px * self.opt.width
        # world-frame values at the home pose (constant only in fixed_sensor mode)
        self.gel_top_z = float(self.data.geom_xpos[self.gel_id][2] + m.geom_size[self.gel_id][2])
        self.cam_z = float(self.data.cam_xpos[self.cam_id][2])
        self.sensor_xy = self.data.geom_xpos[self.gel_id][:2].copy()

        self._renderer = mujoco.Renderer(m, self.opt.height, self.opt.width)
        self._renderer.enable_depth_rendering()
        self._scene_opt = mujoco.MjvOption()
        self._scene_opt.geomgroup[:] = 0
        self._scene_opt.geomgroup[ACTIVE_GROUP] = 1           # active object only: never table/arm/sensor
        self._scene_opt.sitegroup[:] = 0
        self.render_table = bool(render_table)
        if self.render_table:
            if not self.moving:
                raise ValueError("render_table=True needs mode='moving_sensor'")
            others = [m.geom(g).name for g in range(m.ngeom) if m.geom_group[g] == TABLE_GROUP and g != self.table_id]
            if others:
                raise ValueError(f"render_table: geom group {TABLE_GROUP} must hold only the table, also has {others}")

        self._build_optics()
        active = [t for t in self.objects if m.geom_group[self.tool_geoms[t][0]] == ACTIVE_GROUP]
        first = DEFAULT_OBJECT if DEFAULT_OBJECT in self.objects else (active[0] if active else "fork_dinner")
        if self.moving:
            self.object_top = {t: self._object_top(t) for t in self.objects}
            self.place_object(first)
        else:
            self.set_tool(first)

    def _camera_to_gel(self) -> float:
        m = self.model
        if m.cam_bodyid[self.cam_id] != m.geom_bodyid[self.gel_id]:
            raise ValueError("gelsight_cam and the gel must belong to the same body")
        g, c = self.gel_id, self.cam_id
        top = m.geom_pos[g] + _quat_to_mat(m.geom_quat[g]) @ np.array([0.0, 0.0, m.geom_size[g][2]])
        view = -_quat_to_mat(m.cam_quat[c])[:, 2]          # cameras look along their -z
        return float(np.dot(top - m.cam_pos[c], view))

    # ------------------------------------------------------------------ tools / pose
    def set_tool(self, name: str):
        """Make `name` the only active object (fixed_sensor: the tool on the indenter;
        moving_sensor: the object on the table, left wherever it was last placed)."""
        if name not in self.tool_geoms:
            raise ValueError(f"unknown tool {name!r} in this model; choose from {self.objects}")
        m = self.model
        for t, geoms in self.tool_geoms.items():
            on = t == name
            for g in geoms:
                # mesh objects: the visual mesh never collides, the convex parts are never drawn
                if g not in self.collision_geoms:
                    m.geom_group[g] = ACTIVE_GROUP if on else PARKED_GROUP
                collide = on and g not in self.visual_geoms
                m.geom_contype[g] = 1 if collide else 0
                m.geom_conaffinity[g] = 1 if collide else 0
            # MuJoCo also caches the OR of geom masks per body for the broad phase
            bid = m.body(t).id
            m.body_contype[bid] = 1 if on else 0
            m.body_conaffinity[bid] = 1 if on else 0
            if self.free_objects:
                # parked objects float in place (no gravity, no contacts); the active one falls
                m.body_gravcomp[bid] = 0.0 if on else 1.0
                # MuJoCo scales soft-contact compliance by the invweight0 of BOTH bodies in
                # contact, so a light free object would make the gel up to ~3x softer than
                # a static one (and let it sink into the table). A small value keeps the
                # gel's force -> indentation law a property of the sensor (within ~2 % of
                # the static case) and the object-table contact stiff. Exactly 0 makes
                # that contact rigid, which the solver cannot handle (it blows up).
                m.body_invweight0[bid] = (FREE_INVWEIGHT_FRAC * self._invweight0[self.sensor_body]
                                          if on else self._invweight0[bid])
                d = self.data
                v = self._obj_dadr[t]
                d.qvel[v:v + 6] = 0.0
        self.tool = name

    def place_object(self, name: str, x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
        """moving_sensor: make `name` the active object and lay it on the table at (x, y) [m],
        turned by `yaw` [rad] about the vertical. Which face is up and how high the body
        sits come from its resting pose in the XML."""
        if not self.moving:
            raise RuntimeError("place_object() needs mode='moving_sensor'; use set_tool()")
        self.set_tool(name)
        m, d = self.model, self.data
        qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, qz, self.rest_quat[name])
        pos = np.array([x, y, 0.0]) + _quat_to_mat(qz) @ self.rest_pos[name]
        if self.free_objects:
            a, v = self._obj_qadr[name], self._obj_dadr[name]
            d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = pos, quat
            d.qvel[v:v + 6] = 0.0
        else:
            b = self.tool_body[name]
            m.body_pos[b], m.body_quat[b] = pos, quat
        self.object_pose[name] = dict(x=float(x), y=float(y), yaw=float(yaw))
        mujoco.mj_forward(m, d)

    def set_rest_pose(self, name: str, pos, quat):
        """moving_sensor: the resting pose place_object() uses for `name` from now on: body
        position at the table origin [m] and orientation quaternion (w, x, y, z), e.g. another
        stable pose than the XML's. Takes effect at the next place_object()."""
        if not self.moving:
            raise RuntimeError("set_rest_pose() needs mode='moving_sensor'")
        q = np.asarray(quat, float)
        self.rest_pos[name] = np.asarray(pos, float).copy()
        self.rest_quat[name] = q / np.linalg.norm(q)
        self.object_top[name] = self._object_top(name)

    def sensor_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """World pose of the sensor frame: origin at the centre of the undeformed gel contact
        face [m]; axes (columns of R) of the body "gelsight": +x = image column, +y = image row,
        +z = viewing direction, i.e. out of the gel face into the object (world down for an
        untilted moving sensor). Pixel (col, row) <-> sensor (x, y): see px_to_sensor_mm()."""
        m, d, g = self.model, self.data, self.gel_id
        R_gel = d.geom_xmat[g].reshape(3, 3)
        origin = d.geom_xpos[g] + R_gel @ np.array([0.0, 0.0, m.geom_size[g][2]])
        return origin.copy(), d.xmat[self.sensor_body].reshape(3, 3).copy()

    def _object_top(self, name: str) -> float:
        """Height of the highest point of `name` lying on the table (from geom bounding boxes)."""
        m, d = self.model, self.data
        top = 0.0
        b = self.tool_body[name]
        R_b = _quat_to_mat(self.rest_quat[name])
        corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        for g in self.tool_geoms[name]:
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            p = (corners * h + c) @ _quat_to_mat(m.geom_quat[g]).T + m.geom_pos[g]
            top = max(top, float((p @ R_b.T + self.rest_pos[name])[:, 2].max()))
        return top

    def sensor_pose_for(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0,
                        pitch: float = 0.0, roll: float = 0.0, site: str | None = None) -> dict:
        """moving_sensor: arm pose (x, y, yaw, pitch, roll) that gives the active object the
        relative pose press(x, y, yaw, pitch, roll) means in fixed_sensor mode: the object's
        tool frame (site "<name>_face": centre of its gel-facing face, +x to the tips, +z into
        the object) at (x, y) [m] in the sensor frame, turned by `yaw` and tilted by `pitch`
        (about its own y) and `roll` (its x). Both modes then produce the same tactile image.
        site: use another surface frame of the object instead, e.g. "fork_stl_handle"."""
        d = self.data
        s = self.face_site[self.tool] if site is None else self.model.site(site).id
        if self.model.site_bodyid[s] != self.tool_body[self.tool]:
            raise ValueError(f"site {site!r} does not belong to the active object {self.tool!r}")
        R_face, p_face = d.site_xmat[s].reshape(3, 3), d.site_xpos[s]
        cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
        R_rel = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                 @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))     # same yaw-pitch-roll chain as the indenter
        S = R_face @ R_rel.T                       # sensor (gelsight body) orientation in the world
        M = S @ np.diag([1.0, -1.0, -1.0])         # wrist = sensor turned back by 180 deg about x
        c = p_face - S @ np.array([x, y, 0.0])     # gel-face centre = arm x, y
        return dict(x=float(c[0]), y=float(c[1]), yaw=float(np.arctan2(M[1, 0], M[0, 0])),
                    pitch=float(np.arcsin(np.clip(-M[2, 0], -1, 1))), roll=float(np.arctan2(M[2, 1], M[2, 2])))

    def set_pose(self, x=0.0, y=0.0, z=None, yaw=0.0, pitch=0.0, roll=0.0):
        """Kinematically place the tool (fixed_sensor, relative to the gel centre) or the sensor
        (moving_sensor, world frame; z = height of the gel face above the table). Metres /
        radians; z defaults to the home height."""
        if z is None:
            z = self.z_home
        d, j, a = self.data, self.jnt, self.act
        d.qpos[j["slide_x"]], d.qpos[j["slide_y"]], d.qpos[j["slide_z"]] = x, y, z
        d.qpos[j["yaw"]], d.qpos[j["pitch"]], d.qpos[j["roll"]] = yaw, pitch, roll
        d.qvel[:] = 0
        d.ctrl[a["x"]], d.ctrl[a["y"]] = x, y
        d.ctrl[a["yaw"]], d.ctrl[a["pitch"]], d.ctrl[a["roll"]] = yaw, pitch, roll
        d.ctrl[a["press"]] = 0.0
        mujoco.mj_forward(self.model, d)

    def _is_sensor_contact(self, c) -> bool:
        return c.geom1 in self.sensor_geoms or c.geom2 in self.sensor_geoms

    def _in_contact(self, z) -> bool:
        d = self.data
        d.qpos[self.jnt["slide_z"]] = z
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_collision(self.model, d)
        if not self.free_objects:
            return d.ncon > 0          # only the moving part can touch anything
        return any(self._is_sensor_contact(d.contact[i]) for i in range(d.ncon))

    def contact_height(self, x=0.0, y=0.0, yaw=0.0, pitch=0.0, roll=0.0) -> float:
        """slide_z at which the moving part first touches something (bisection), for this pose."""
        self.set_pose(x, y, self.z_start, yaw, pitch, roll)
        lo, hi = self.z_range[0], self.z_start
        if not self._in_contact(lo):
            return float(lo)
        for _ in range(30):                      # bisection for the first-contact height
            mid = 0.5 * (lo + hi)
            if self._in_contact(mid):
                lo = mid
            else:
                hi = mid
        return float(hi)

    def press(self, force: float, x=0.0, y=0.0, yaw=0.0, pitch=0.0, roll=0.0,
              max_time: float = 0.6, tol: float = 2e-4) -> PressInfo:
        """Bring the moving part down to first contact, then push with `force` newtons until
        static. fixed_sensor: moves the TOOL (pose relative to the gel centre).
        moving_sensor: moves the SENSOR (arm joints, world frame) onto the object."""
        m, d = self.model, self.data
        com0 = self.object_com() if self.free_objects else None
        z_c = self.contact_height(x, y, yaw, pitch, roll)
        d.qpos[self.jnt["slide_z"]] = z_c + 1e-4
        d.qvel[:] = 0
        d.ctrl[self.act["press"]] = force
        mujoco.mj_forward(m, d)
        steps, settled = 0, False
        nmax = int(max_time / m.opt.timestep)
        while steps < nmax:
            mujoco.mj_step(m, d)
            steps += 1
            if steps > 100 and steps % 20 == 0 and abs(d.qvel[self.dof_z]) < tol and self._object_still(tol):
                settled = True
                break
        mujoco.mj_forward(m, d)
        h = self.indentation_mm(raw=True)
        table, housing = self.sensor_contacts()
        return PressInfo(tool=self.tool, force_cmd=force,
                         gel_force=float(d.sensordata[self.touch_adr]),
                         max_indent_mm=float(h.max()), contact_height_m=float(z_c),
                         settled=settled, steps=steps,
                         visible_tines=self.visible_tines(),
                         pose=dict(x=x, y=y, yaw=yaw, pitch=pitch, roll=roll),
                         object_pose=dict(self.object_pose[self.tool]) if self.moving else {},
                         table_contact=table, housing_contact=housing,
                         object_shift_mm=float(np.linalg.norm(self.object_com() - com0) * 1000)
                         if self.free_objects else 0.0)

    def press_relative(self, force: float, x=0.0, y=0.0, yaw=0.0, pitch=0.0, roll=0.0,
                       site: str | None = None, **kw) -> PressInfo:
        """Press with the active object at the relative pose (x, y, yaw, pitch, roll) - the
        meaning press() has in fixed_sensor mode - in either mode. moving_sensor only: `site`
        picks another surface frame than "<name>_face" (see sensor_pose_for)."""
        if not self.moving:
            if site is not None:
                raise ValueError("site= needs mode='moving_sensor'")
            return self.press(force, x, y, yaw, pitch, roll, **kw)
        return self.press(force, **self.sensor_pose_for(x, y, yaw, pitch, roll, site=site), **kw)

    def _object_still(self, tol: float) -> bool:
        """free_objects: the active object's centre of mass is (nearly) still. Under a tilted
        press MuJoCo's soft friction lets a free object creep slowly out of the gel/table
        pinch; creep below FREE_CREEP_TOL (< 1 px per 100 ms) counts as still. Angular
        velocity is not checked: a ball pinched between gel and table may spin in place."""
        if not self.free_objects:
            return True
        v = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.tool_body[self.tool], v, 0)
        return bool(np.abs(v[3:]).max() < max(tol, FREE_CREEP_TOL))

    def object_com(self) -> np.ndarray:
        """World position of the active object's centre of mass [m]."""
        return self.data.xipos[self.tool_body[self.tool]].copy()

    def sensor_contacts(self) -> tuple[bool, bool]:
        """(table_contact, housing_contact) in the current state: the sensor touches the
        table / a sensor part other than the gel touches something."""
        d = self.data
        table = housing = False
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in self.sensor_geoms:
                mine, other = c.geom1, c.geom2
            elif c.geom2 in self.sensor_geoms:
                mine, other = c.geom2, c.geom1
            else:
                continue
            table |= other == self.table_id
            housing |= mine != self.gel_id
        return table, housing

    def lift(self, height=None):
        """Raise the moving part to `height` [m] (default: home height), keeping x, y, orientation."""
        q = self.data.qpos
        self.set_pose(q[self.jnt["slide_x"]], q[self.jnt["slide_y"]], height,
                      q[self.jnt["yaw"]], q[self.jnt["pitch"]], q[self.jnt["roll"]])

    def visible_tines(self, min_depth_mm: float = 0.05) -> int:
        """Ground truth: number of distinct fork tines pressed at least `min_depth_mm`
        into the gel inside the camera FOV."""
        if self.tool not in FORKS:
            return 0
        m, d = self.model, self.data
        tines = set()
        hw, hh = self.fov_w_mm / 2000.0, self.fov_h_mm / 2000.0
        # field of view test in the camera frame (x = image right, y = image up): works
        # wherever the sensor is and however it is turned
        R, p0 = d.cam_xmat[self.cam_id].reshape(3, 3), d.cam_xpos[self.cam_id]
        for i in range(d.ncon):
            c = d.contact[i]
            if self.gel_id not in (c.geom1, c.geom2):
                continue                         # free objects also touch the table
            g = c.geom2 if c.geom1 == self.gel_id else c.geom1
            name = m.geom(g).name
            if name.startswith(self.tool) and ("_tine" in name or "_tip" in name):
                p = R.T @ (c.pos - p0)
                if abs(p[0]) < hw and abs(p[1]) < hh and c.dist < -min_depth_mm / 1000.0:
                    tines.add(name[-1])
        return len(tines)

    # ------------------------------------------------------------------ images
    def indentation_mm(self, raw: bool = False, table: bool | None = None) -> np.ndarray:
        """Object penetration past the undeformed gel surface, in mm (H x W).
        raw=True: rigid object footprint; raw=False: deformed gel surface (with skirt).
        table: also draw the table (default: the render_table setting)."""
        self._scene_opt.geomgroup[TABLE_GROUP] = int(self.render_table if table is None else table)
        self._renderer.update_scene(self.data, camera=self.cam_id, scene_option=self._scene_opt)
        depth = self._renderer.render().astype(np.float32)
        # 3x3 median removes single-pixel depth outliers that appear where box side
        # faces are seen exactly edge-on by the orthographic camera
        depth = cv2.medianBlur(depth, 3).astype(np.float64)
        h = np.clip((self.d0 - depth) * 1000.0, 0.0, self.opt.max_indent_mm)
        return h if raw else self.gel_surface(h)

    def gel_surface(self, h_obj: np.ndarray) -> np.ndarray:
        o, s = self.opt, self.mm_per_px
        skirt = o.spread_gain * gaussian_filter(h_obj, o.spread_sigma_mm / s)
        h = np.maximum(h_obj, skirt)
        return gaussian_filter(h, o.edge_sigma_mm / s)

    def _build_optics(self):
        o = self.opt
        H, W = o.height, o.width
        az = np.deg2rad(np.asarray(o.led_azimuth_deg))
        el = np.deg2rad(o.led_elevation_deg)
        self.light_dirs = np.stack([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el),
                                    np.full(3, np.sin(el))], axis=1)          # (3 LEDs, xyz)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        u = (xx - (W - 1) / 2) / (W / 2)
        v = (yy - (H - 1) / 2) / (H / 2)
        bg = np.empty((H, W, 3))
        for c in range(3):
            lx, ly = self.light_dirs[c, :2] / np.linalg.norm(self.light_dirs[c, :2])
            bg[..., c] = o.background_rgb[c] * (1.0 + o.led_falloff * (u * lx + v * ly))
        self._bg = bg
        # marker rest positions (pixel coords)
        pitch_px = o.marker_pitch_mm / self.mm_per_px
        nx, ny = int(W / pitch_px), int(H / pitch_px)
        mx = (np.arange(nx) - (nx - 1) / 2) * pitch_px + (W - 1) / 2
        my = (np.arange(ny) - (ny - 1) / 2) * pitch_px + (H - 1) / 2
        self._markers = np.stack(np.meshgrid(mx, my), -1).reshape(-1, 2)   # (N, [col,row])

    def shade(self, h_gel: np.ndarray, noise: bool = True) -> np.ndarray:
        """Photometric model: RGB image of a gel surface with indentation h_gel (mm)."""
        o, s = self.opt, self.mm_per_px
        gy, gx = np.gradient(h_gel, s)                      # slopes (row = +y, col = +x)
        n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        shading = n @ self.light_dirs.T - self.light_dirs[:, 2]     # 0 on flat gel
        img = self._bg + shading * np.asarray(o.led_gain)
        if o.markers:
            img *= (1.0 - o.marker_darkness * self._marker_mask(h_gel))[..., None]
        if o.optics_blur_px > 0:
            img = cv2.GaussianBlur(img, (0, 0), o.optics_blur_px)
        if noise and o.noise_std > 0:
            img = img + self.rng.normal(0, o.noise_std, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def marker_field(self, h_gel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Markers for a gel surface h_gel (mm): rest positions and displacements, both (N, 2)
        in pixels as (column, row); a marker is drawn at rest + displacement."""
        o, s = self.opt, self.mm_per_px
        H, W = h_gel.shape
        sm = gaussian_filter(h_gel, 1.0 / s)
        gy, gx = np.gradient(sm, s)
        pts = self._markers
        ci = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
        ri = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
        shift = -o.marker_shift_gain_mm / s * np.stack([gx[ri, ci], gy[ri, ci]], -1)  # outward
        return pts.copy(), shift

    def _marker_mask(self, h_gel: np.ndarray) -> np.ndarray:
        o, s = self.opt, self.mm_per_px
        H, W = h_gel.shape
        pts, shift = self.marker_field(h_gel)
        p = pts + shift
        mask = np.zeros((H, W), np.uint8)
        r = o.marker_radius_mm / s
        sh = 4                                              # sub-pixel drawing
        for (cx, cy) in p:
            cv2.circle(mask, (int(round(cx * 2**sh)), int(round(cy * 2**sh))),
                       int(round(r * 2**sh)), 255, -1, cv2.LINE_AA, sh)
        return mask.astype(np.float64) / 255.0

    def tactile_image(self, noise: bool = True) -> np.ndarray:
        return self.shade(self.indentation_mm(), noise=noise)

    def reference_image(self, noise: bool = True) -> np.ndarray:
        return self.shade(np.zeros((self.opt.height, self.opt.width)), noise=noise)

    # ------------------------------------------------------------------ helpers
    def px_to_sensor_mm(self, col, row):
        """Image pixel -> (x, y) in mm on the gel in the SENSOR frame, relative to the gel
        centre (image column = sensor +x, image row = sensor +y, in both modes)."""
        x = (np.asarray(col) - (self.opt.width - 1) / 2) * self.mm_per_px
        y = (np.asarray(row) - (self.opt.height - 1) / 2) * self.mm_per_px
        return x, y

    def px_to_world_mm(self, col, row):
        """Image pixel -> world (x, y) in mm of that point of the undeformed gel surface, from
        the current camera pose. fixed_sensor: the gel centre is the world origin, so this
        equals px_to_sensor_mm(); moving_sensor: follows the sensor's position, yaw and tilt."""
        u, v = self.px_to_sensor_mm(col, row)              # image right, image down [mm]
        R = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        p = self.data.cam_xpos[self.cam_id] * 1000.0
        w = self.d0 * 1000.0
        # camera frame: x = image right, y = image up, the gel lies at z = -d0
        x = p[0] + R[0, 0] * u - R[0, 1] * v - R[0, 2] * w
        y = p[1] + R[1, 0] * u - R[1, 1] * v - R[1, 2] * w
        return x, y

    def image_angle_to_world(self, angle_rad: float) -> float:
        """Direction in the tactile image (radians from +column towards +row, as reported by
        ForkDetector) -> heading in the world xy plane."""
        R = self.data.cam_xmat[self.cam_id].reshape(3, 3)
        v = R @ np.array([np.cos(angle_rad), -np.sin(angle_rad), 0.0])
        return float(np.arctan2(v[1], v[0]))

    def render_scene(self, width=960, height=720, azimuth=None, elevation=None,
                     distance=None, lookat=None):
        """Pretty third-person render of the sensor + object (for figures). In moving_sensor
        mode the default view follows the sensor: centred on it, looking through its
        see-through front wall at the camera, LEDs and gel on the object."""
        if self.moving:
            gx, gy = self.data.geom_xpos[self.gel_id][:2]
            psi = np.degrees(self.data.qpos[self.jnt["yaw"]])
            dflt = dict(azimuth=psi - 100, elevation=-12, distance=0.24, lookat=(gx, gy, 0.035))
        else:
            dflt = dict(azimuth=135, elevation=-28, distance=0.17, lookat=(0, 0, 0.03))
        azimuth = dflt["azimuth"] if azimuth is None else azimuth
        elevation = dflt["elevation"] if elevation is None else elevation
        distance = dflt["distance"] if distance is None else distance
        lookat = dflt["lookat"] if lookat is None else lookat
        r = mujoco.Renderer(self.model, height, width)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
        cam.lookat[:] = lookat
        opt = mujoco.MjvOption()
        r.update_scene(self.data, camera=cam, scene_option=opt)
        img = r.render()
        r.close()
        return img

    def close(self):
        self._renderer.close()
