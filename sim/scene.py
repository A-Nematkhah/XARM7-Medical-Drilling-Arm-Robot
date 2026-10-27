"""
Compose the full scene with MjSpec (MuJoCo >= 3.2):

    world ─ floor/table, lights, workpiece board
          ├─ xArm7 (MuJoCo Menagerie, no gripper)
          │    └─ link7/attachment_site ─ adapter flange ─ Inspire RH56E2 (right)
          │                                             └─ Intel RealSense D435i (wrist bracket)
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


def _quat_from_z(yaw):
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def build_spec(drill_xy=(0.47, -0.02), drill_yaw=0.0):
    if not os.path.isfile(HAND_XML):
        import runpy
        runpy.run_path(os.path.join(ROOT, "models", "build_hand.py"), run_name="__main__")

    spec = mujoco.MjSpec()
    spec.modelname = "xarm7_inspire_drill"
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10          # stiffer friction -> less slip in grasps
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
    wb.add_light(pos=[0.3, -0.4, 1.6], dir=[0, 0.25, -1], diffuse=[0.7, 0.7, 0.7], castshadow=True)
    wb.add_light(pos=[1.2, 0.8, 1.2], dir=[-0.6, -0.4, -0.6], diffuse=[0.35, 0.35, 0.35], castshadow=False)
    wb.add_camera(name="overview", pos=[1.35, -0.95, 0.85], xyaxes=[0.58, 0.81, 0, -0.33, 0.24, 0.91])

    board = wb.add_body(name="board", pos=BOARD_POS.tolist())
    board.add_geom(name="board", type=mujoco.mjtGeom.mjGEOM_BOX, size=BOARD_HALF.tolist(),
                   rgba=[0.80, 0.64, 0.42, 1])
    for s in (-1, 1):  # two stands near the face edges
        board.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.06, 0.012, 0.012],
                       pos=[0.0, s * 0.75 * BOARD_HALF[1], -BOARD_HALF[2]], rgba=[0.3, 0.3, 0.3, 1])
    board.add_geom(name="board_leg", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, (BOARD_POS[2] - BOARD_HALF[2]) / 2],
                   pos=[0.03, 0, -BOARD_HALF[2] - (BOARD_POS[2] - BOARD_HALF[2]) / 2], rgba=[0.3, 0.3, 0.3, 1])
    # hole marker grown at runtime while drilling (visual only)
    board.add_geom(name="hole", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0001, 0.0001, 0],
                   pos=[-BOARD_HALF[0] - 0.0005, 0, 0], quat=[0.7071068, 0, 0.7071068, 0],
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

    # ---------------- drill ----------------
    drill = mujoco.MjSpec.from_file(DRILL_XML)
    dframe = wb.add_frame(pos=[drill_xy[0], drill_xy[1], 0.0005], quat=_quat_from_z(drill_yaw).tolist())
    dframe.attach_body(drill.body("drill"), "", "")

    # (attach_body also copies actuators, equality constraints and contact excludes)
    # the spinning bit/chuck may pass into the board: material removal is modelled
    # kinematically by the task (hole depth + marker), not by rigid contact
    spec.add_exclude(bodyname1="chuck", bodyname2="board")
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
