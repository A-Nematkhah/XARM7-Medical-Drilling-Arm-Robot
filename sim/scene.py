"""
Compose the full scene with MjSpec (MuJoCo >= 3.2):

    world ─ floor/table, lights, workpiece board
          ├─ xArm7 (MuJoCo Menagerie, no gripper)
          │    └─ link7/attachment_site ─ adapter flange ─ Inspire RH56E2 (right)
          │                                             ├─ Intel RealSense D435i (wrist bracket)
          │                                             └─ 360° LiDAR puck (pinky side)
          └─ drill (free body, randomizable pose)
"""
import os
import numpy as np
import mujoco

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARM_XML = os.path.join(ROOT, "assets", "xarm7", "xarm7_nohand.xml")
HAND_XML = os.path.join(ROOT, "models", "inspire_rh56e2_right.xml")
DRILL_XML = os.path.join(ROOT, "models", "drill.xml")

# ---- D435i color stream (1280x720 16:9, VFOV ~42 deg). Rendered at 640x360. ----
CAM_W, CAM_H, CAM_FOVY = 640, 360, 42.0
# Camera pose in the hand 'rh_base' frame (+X palm normal, +Y index side, +Z fingers).
# Optical axis looks along +X (out of the palm), image-up = +Y (index side).
CAM_POS = (0.004, 0.072, 0.030)

# Workpiece: a vertical plywood board the robot will drill into.
# Face area is 1.5x the original (Y/Z scaled by sqrt(1.5)); raised for easier angled drills.
_BOARD_FACE_SCALE = np.sqrt(1.5)
BOARD_POS = np.array([0.64, -0.34, 0.34])
BOARD_HALF = np.array([0.012, 0.13 * _BOARD_FACE_SCALE, 0.11 * _BOARD_FACE_SCALE])
# High-contrast hole target on the face (board-frame Y/Z offsets from centre).
HOLE_YZ = np.array([0.02, -0.01])
# Pad half-size 18 mm; keep the disk inside the face with wood around it.
MARKER_INSET = 0.030
# Sim spawn keeps BOARD_POS / yaw=0 fixed. Each run jitters the face mark
# (hole_yz) and the drill XY in a patch left of the board (+Y). Board +Y edge
# is ~-0.18 m.
DRILL_SPAWN_LO = np.array([0.30, -0.14])
DRILL_SPAWN_HI = np.array([0.58, 0.28])
DRILL_YAW_LIM_DEG = 25.0
MIN_DRILL_BOARD_XY = 0.18
# Extra standoff when the drill sits in front of the face (side-slide grasp).
MIN_DRILL_FACE_XY = 0.24
# Wrist LiDAR origin in rh_base (must match config/calib.yaml lidar_eye via TCP).
# Pinky side of the palm; body Rx(180) so the spin axis is world-up at REST_Q.
LIDAR_T = np.array([0.02, -0.078, 0.02])
LIDAR_QUAT = np.array([0.0, 1.0, 0.0, 0.0])  # Rx(180)
LIDAR_RPY_DEG = np.array([180.0, 0.0, 0.0])


def _quat_from_z(yaw):
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def _R_from_z(yaw):
    c, s = np.cos(float(yaw)), np.sin(float(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def hole_on_face(board_pos=None, hole_yz=None, board_yaw=0.0):
    """World pierce point and into-board normal for a spawned board (eval GT)."""
    p = BOARD_POS if board_pos is None else np.asarray(board_pos, float)
    hyz = HOLE_YZ if hole_yz is None else np.asarray(hole_yz, float)
    R = _R_from_z(board_yaw)
    local = np.array([-BOARD_HALF[0], float(hyz[0]), float(hyz[1])])
    n = R @ np.array([1.0, 0.0, 0.0])
    return p + R @ local, n / (np.linalg.norm(n) + 1e-12)


def sample_board(rng):
    """Fixed board pose/yaw; random hole_yz on the face (inside the marker inset)."""
    hy_m = float(BOARD_HALF[1] - MARKER_INSET)
    hz_m = float(BOARD_HALF[2] - MARKER_INSET)
    hole_yz = rng.uniform([-hy_m, -hz_m], [hy_m, hz_m])
    return BOARD_POS.copy(), hole_yz, 0.0


def board_left_y():
    """World +Y of the board's left edge (robot looking at the face)."""
    return float(BOARD_POS[1] + BOARD_HALF[1])


def drill_spawn_center():
    return 0.5 * (DRILL_SPAWN_LO + DRILL_SPAWN_HI)


def drill_spawn_half():
    return 0.5 * (DRILL_SPAWN_HI - DRILL_SPAWN_LO)


def drill_clear_of_board(xy, board_pos, board_yaw=0.0):
    """True if table XY is far enough from the board centre and the face slab."""
    xy = np.asarray(xy, float).reshape(2)
    pos = np.asarray(board_pos, float).reshape(3)
    if float(np.linalg.norm(xy - pos[:2])) < MIN_DRILL_BOARD_XY:
        return False
    R = _R_from_z(board_yaw)
    n_xy = (R @ np.array([1.0, 0.0, 0.0]))[:2]
    n_xy = n_xy / (np.linalg.norm(n_xy) + 1e-12)
    t_xy = (R @ np.array([0.0, 1.0, 0.0]))[:2]
    t_xy = t_xy / (np.linalg.norm(t_xy) + 1e-12)
    face = (pos + R @ np.array([-BOARD_HALF[0], 0.0, 0.0]))[:2]
    rel = xy - face
    along = float(rel @ n_xy)  # >0 = into / behind the face
    lat = abs(float(rel @ t_xy))
    in_front_corridor = along > -MIN_DRILL_FACE_XY and lat <= float(BOARD_HALF[1]) + 0.08
    return not in_front_corridor


def sample_drill_xy(rng, board_pos=None, board_yaw=0.0, n_try=120):
    """Table XY in the left-of-board spawn patch. Extra args kept for call sites."""
    _ = (board_pos, board_yaw, n_try)
    return rng.uniform(DRILL_SPAWN_LO, DRILL_SPAWN_HI)


def sample_layout(rng, n_try=40):
    """Fixed board; random face mark + drill XY in the left-of-board patch."""
    _ = n_try
    pos, hole_yz, board_yaw = sample_board(rng)
    xy = sample_drill_xy(rng, pos, board_yaw)
    drill_yaw = np.deg2rad(float(rng.uniform(-DRILL_YAW_LIM_DEG, DRILL_YAW_LIM_DEG)))
    return xy, drill_yaw, pos, hole_yz, board_yaw


HOLE_TARGET, BOARD_N = hole_on_face()


def optical_look_at_R(pos, look, up=(0.0, 0.0, 1.0)):
    """Rotation whose optical frame looks from `pos` toward `look` (-Z, +Y up)."""
    pos = np.asarray(pos, float)
    f = np.asarray(look, float) - pos
    f = f / (np.linalg.norm(f) + 1e-12)
    up = np.asarray(up, float)
    z = -f
    x = np.cross(up, f)
    xn = np.linalg.norm(x)
    if xn < 1e-9:
        x = np.array([0.0, 1.0, 0.0])
    else:
        x = x / xn
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-12)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-12)
    return np.column_stack([x, y, z])


def _mat_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def build_spec(drill_xy=(0.47, -0.02), drill_yaw=0.0,
               board_pos=None, hole_yz=None, board_yaw=0.0):
    if not os.path.isfile(HAND_XML):
        import runpy
        runpy.run_path(os.path.join(ROOT, "models", "build_hand.py"), run_name="__main__")

    spec = mujoco.MjSpec()
    spec.modelname = "xarm7_inspire_drill"
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 20          # stiffer friction -> less slip under drill thrust
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    spec.stat.center = [0.35, 0, 0.25]
    spec.stat.extent = 0.9

    # ---------------- environment ----------------
    wb = spec.worldbody
    tex = spec.add_texture(name="wood", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.46, 0.36, 0.26], rgb2=[0.42, 0.33, 0.24], width=512, height=512)
    mat = spec.add_material(name="table")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "wood"
    mat.texrepeat = [6, 6]
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.2, 1.2, 0.05],
                material="table", friction=[1.0, 0.01, 0.001])
    # Visual-only patch: `--random` drill XY stays inside this left-of-board box.
    c = drill_spawn_center()
    h = drill_spawn_half()
    wb.add_geom(name="drill_spawn_box", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[float(h[0]), float(h[1]), 0.0015],
                pos=[float(c[0]), float(c[1]), 0.002],
                rgba=[0.10, 0.75, 0.95, 0.35], contype=0, conaffinity=0)
    edge_rgba = [0.05, 0.95, 1.0, 0.95]
    for dx, dy, sx, sy in (
            (0.0, h[1], h[0] + 0.004, 0.004),
            (0.0, -h[1], h[0] + 0.004, 0.004),
            (h[0], 0.0, 0.004, h[1]),
            (-h[0], 0.0, 0.004, h[1]),
    ):
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[float(sx), float(sy), 0.0025],
                    pos=[float(c[0] + dx), float(c[1] + dy), 0.004],
                    rgba=edge_rgba, contype=0, conaffinity=0)
    wb.add_light(pos=[0.3, -0.4, 1.6], dir=[0, 0.25, -1], diffuse=[0.7, 0.7, 0.7], castshadow=True)
    wb.add_light(pos=[1.2, 0.8, 1.2], dir=[-0.6, -0.4, -0.6], diffuse=[0.35, 0.35, 0.35], castshadow=False)
    wb.add_camera(name="overview", pos=[1.35, -0.95, 0.85], xyaxes=[0.58, 0.81, 0, -0.33, 0.24, 0.91])

    board_pos = BOARD_POS if board_pos is None else np.asarray(board_pos, float).reshape(3)
    hole_yz = HOLE_YZ if hole_yz is None else np.asarray(hole_yz, float).reshape(2)
    board_yaw = float(board_yaw)
    board = wb.add_body(name="board", pos=board_pos.tolist(),
                        quat=_quat_from_z(board_yaw).tolist())
    board.add_geom(name="board", type=mujoco.mjtGeom.mjGEOM_BOX, size=BOARD_HALF.tolist(),
                   rgba=[0.80, 0.64, 0.42, 1])
    for s in (-1, 1):  # two stands near the face edges
        board.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, 0.012, 0.012],
                       pos=[0.0, s * 0.75 * BOARD_HALF[1], -BOARD_HALF[2]], rgba=[0.3, 0.3, 0.3, 1])
    leg_half = max((float(board_pos[2]) - BOARD_HALF[2]) / 2.0, 0.01)
    board.add_geom(name="board_leg", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, leg_half],
                   pos=[0.03, 0, -BOARD_HALF[2] - leg_half], rgba=[0.3, 0.3, 0.3, 1])
    # High-contrast pierce target on the face (white pad + black disk) for perception.
    face_x = -BOARD_HALF[0] - 0.0006
    hy, hz = float(hole_yz[0]), float(hole_yz[1])
    board.add_geom(name="target_pad", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.0005, 0.018, 0.018], pos=[face_x, hy, hz],
                   rgba=[0.95, 0.95, 0.95, 1], contype=0, conaffinity=0)
    board.add_geom(name="target_dot", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[0.008, 0.00055], pos=[face_x - 0.0003, hy, hz],
                   quat=[0.7071068, 0, 0.7071068, 0],
                   rgba=[0.04, 0.04, 0.04, 1], contype=0, conaffinity=0)
    # hole marker grown at runtime while drilling (visual only)
    board.add_geom(name="hole", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0001, 0.0001, 0],
                   pos=[-BOARD_HALF[0] - 0.0005, hy, hz], quat=[0.7071068, 0, 0.7071068, 0],
                   rgba=[0.12, 0.08, 0.05, 0], contype=0, conaffinity=0)

    # ---------------- xArm7 ----------------
    arm = mujoco.MjSpec.from_file(ARM_XML)
    arm.meshdir = os.path.join(ROOT, "assets", "xarm7", "assets")
    for b in arm.bodies:
        b.gravcomp = 1.0      # like the xArm controller's gravity compensation
    arm_frame = wb.add_frame(pos=[0, 0, 0])
    arm_frame.attach_body(arm.body("link_base"), "", "")

    # ---------------- Inspire RH56E2 ----------------
    hand = mujoco.MjSpec.from_file(HAND_XML)
    for b in hand.bodies:
        b.gravcomp = 1.0
    link7 = spec.body("link7")
    flange = link7.add_body(name="hand_adapter", pos=[0, 0, 0])
    flange.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.031, 0.006], pos=[0, 0, 0.006],
                    rgba=[0.2, 0.2, 0.22, 1], mass=0.05)
    hframe = flange.add_frame(pos=[0, 0, 0.012])
    hframe.attach_body(hand.body("rh_base"), "", "")
    rh_base = spec.body("rh_base")

    # ---------------- RealSense D435i on a wrist bracket ----------------
    cam_body = rh_base.add_body(name="d435i", pos=list(CAM_POS))
    cam_body.add_geom(name="d435i_case", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0125, 0.0125, 0.045],
                      rgba=[0.12, 0.12, 0.13, 1], mass=0.072, contype=0, conaffinity=0)
    for dz in (-0.0325, -0.0125, 0.0325):  # IR / RGB lenses
        cam_body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.004, 0.001], pos=[0.0126, 0, dz],
                          quat=[0.7071068, 0, 0.7071068, 0], rgba=[0.02, 0.02, 0.05, 1], contype=0, conaffinity=0)
    cam_body.add_geom(name="d435i_bracket", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.004, 0.02, 0.01],
                      pos=[-0.012, -0.032, -0.02], rgba=[0.5, 0.5, 0.52, 1], contype=0, conaffinity=0)
    # MuJoCo cameras look along their local -Z with +Y up:  -Z_cam = +X_hand, +Y_cam = +Y_hand
    cam_body.add_camera(name="d435i_color", pos=[0.0126, 0, -0.0125], fovy=CAM_FOVY,
                        xyaxes=[0, 0, 1, 0, 1, 0])

    # 360° LiDAR puck on the pinky side. Live origin = TCP FK × lidar_eye, not this geom.
    lidar = rh_base.add_body(name="wrist_lidar", pos=LIDAR_T.tolist(),
                             quat=LIDAR_QUAT.tolist())
    lidar.add_geom(name="wrist_lidar_puck", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[0.022, 0.014], rgba=[0.08, 0.08, 0.1, 1], mass=0.04,
                   contype=0, conaffinity=0)
    rh_base.add_geom(name="wrist_lidar_bracket", type=mujoco.mjtGeom.mjGEOM_BOX,
                     size=[0.006, 0.028, 0.006], pos=[0.012, -0.040, 0.012],
                     rgba=[0.5, 0.5, 0.52, 1], mass=0.01, contype=0, conaffinity=0)

    # ---------------- drill ----------------
    drill = mujoco.MjSpec.from_file(DRILL_XML)
    dframe = wb.add_frame(pos=[drill_xy[0], drill_xy[1], 0.0005], quat=_quat_from_z(drill_yaw).tolist())
    dframe.attach_body(drill.body("drill"), "", "")

    # Bit geoms are contype=0 in drill.xml (flute occupies the hole). Cutting
    # thrust/torque is applied in task/runtime.py via sim/cutting.py.
    # Chuck↔board contact is real — no exclude — so over-feed stops on the chuck.
    return spec


def build_model(**kw):
    spec = build_spec(**kw)
    model = spec.compile()
    return model, spec


if __name__ == "__main__":
    m, s = build_model()
    print(f"nq={m.nq} nv={m.nv} nu={m.nu} neq={m.neq} ncam={m.ncam}")
    print("actuators:", [m.actuator(i).name for i in range(m.nu)])
    with open(os.path.join(ROOT, "outputs", "scene_compiled.xml"), "w") as fh:
        fh.write(s.to_xml())
