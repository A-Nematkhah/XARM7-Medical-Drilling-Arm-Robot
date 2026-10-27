"""
xArm7 + Inspire RH56E2 + wrist RealSense D435i: find a cordless drill, pick it up,
drill a hole in a board, and put the drill back.

    python run_drill_task.py --viewer                 # MuJoCo viewer + live wrist-camera window
    python run_drill_task.py --video outputs/demo.mp4 # headless, records a video
    python run_drill_task.py --random 3               # randomised drill pose (seed 3)
    python run_drill_task.py --viewer --tilt 25 --azim -15   # angled hole (up + left/right)

Pipeline
  1. look down from above  -> coarse drill position + axis          (perception)
  2. side view perpendicular to the drill axis -> grip centre, drill axis,
     housing underside, bit tip                                      (perception)
  3. grasp plan: palm on the back of the grip, index at trigger height;
     approach sideways along the finger direction, thumb pre-opposed
  4. power grasp (little/ring/middle + thumb), lift, force check
  5. look at the board with the wrist camera -> detect face plane + hole
  6. approach along the bit axis using tip-to-board distance (no collision)
  7. pull trigger, feed in, retract
  8. place the drill back where it was and retreat
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "sim"))

ap = argparse.ArgumentParser()
ap.add_argument("--viewer", action="store_true", help="open the interactive viewer + wrist D435i window")
ap.add_argument("--no-wrist-cam", action="store_true",
                help="do not open the live wrist-camera window (viewer still works)")
ap.add_argument("--video", default=None, help="record an mp4 (headless)")
ap.add_argument("--random", type=int, default=None, help="randomise drill pose with this seed")
ap.add_argument("--xy", type=float, nargs=2, default=(0.47, -0.02))
ap.add_argument("--yaw", type=float, default=0.0, help="drill yaw on the table (deg)")
ap.add_argument("--tilt", type=float, default=0.0,
                help="bit tilt from board normal (deg): +up / -down")
ap.add_argument("--azim", type=float, default=0.0,
                help="bit yaw from board normal (deg): + toward +Y / - toward -Y")
ap.add_argument("--no-noise", action="store_true", help="ideal depth (no D435i noise model)")
ap.add_argument("--video-size", type=int, nargs=2, default=(960, 540), help="video width height")
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--no-place", action="store_true", help="skip putting the drill back")
ap.add_argument("--real-robot", action="store_true",
                help="use tighter SafetyGates limits intended for hardware")
args = ap.parse_args()
if not args.viewer:
    os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import scene
from robot import Robot, REST_Q, ViewPlanner, mat2quat
from perception import D435i, DrillDetector, BoardDetector
from safety import SafetyGates, GateError

# ------------------------------------------------------------------ task constants
UP = np.array([0.0, 0.0, 1.0])
GRIP_BELOW_HOUSING = 0.051     # grasp_site height below the housing underside (index on trigger)
SLIDE = 0.16                   # sideways approach distance along the finger direction
STANDOFF = 0.008               # palm-to-grip clearance while sliding in, closed at the end
TRIGGER_ON, TRIGGER_OFF = 1.0e-3, 0.6e-3   # switch hysteresis (m of trigger travel)
MOTOR_SPEED = 150.0            # rad/s (~1400 rpm, typical low gear)
# Nominal board pose is only a search prior for the first look; drilling uses detection.
BOARD_N_HINT = np.array([1.0, 0.0, 0.0])
HOLE_HINT = scene.BOARD_POS + np.array([-scene.BOARD_HALF[0], 0.02, -0.01])
AIM_STANDOFF = 0.03            # desired tip-to-face clearance before trigger (m)
APPROACH_CLEAR = 0.14          # start the distance servo this far from the face (m)
CLEAR_TOL = 0.003              # stop approaching when |clearance - AIM_STANDOFF| < this
DRILL_DEPTH = 0.025
FEED_MARGIN = 0.008            # extra stroke: the drill settles a few mm in the hand when lifted
HAND_GRASP = [0, 0, 0, 1000, 0, 0]          # index stays off the trigger while holding
HAND_TRIGGER = [0, 0, 0, 0, 0, 0]


def _rodrigues(v, axis, deg):
    a = np.deg2rad(deg)
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return (np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)) @ np.asarray(v, float)


def bit_direction(tilt_deg, azim_deg, board_n=BOARD_N_HINT, up=UP):
    """Unit bit axis in world: 0/0 = board normal; +tilt=up, +azim=+Y."""
    n = np.asarray(board_n, float); n = n / np.linalg.norm(n)
    u = np.asarray(up, float); u = u / np.linalg.norm(u)
    d = _rodrigues(n, u, azim_deg)
    tilt_axis = np.cross(d, u)
    if np.linalg.norm(tilt_axis) < 1e-8:
        raise ValueError("tilt undefined when bit is parallel to up")
    d = _rodrigues(d, tilt_axis, tilt_deg)
    return d / np.linalg.norm(d)


def aim_rotation(d, board_n=BOARD_N_HINT, up=UP):
    """Palm along the bit; fingers parallel to the board so they cannot stab into it.

    With a large azim, using world-up as the index axis makes the finger axis
    (d × up) point into the board and the hand collides. Fingers are therefore
    constrained to the board plane (n × d), with the sign that keeps the index
    side toward world-up.
    """
    d = np.asarray(d, float); d = d / np.linalg.norm(d)
    n = np.asarray(board_n, float); n = n / np.linalg.norm(n)
    u = np.asarray(up, float)
    z = np.cross(n, d)                         # // board face
    if np.linalg.norm(z) < 1e-6:
        y = u - d * np.dot(u, d)
        y = y / np.linalg.norm(y)
        return np.column_stack([d, y, np.cross(d, y)])
    z = z / np.linalg.norm(z)
    if np.dot(np.cross(z, d), u) < 0:          # index (y) toward up
        z = -z
    y = np.cross(z, d); y = y / np.linalg.norm(y)
    return np.column_stack([d, y, z])


def quat_align_z(v):
    """Quaternion (wxyz) mapping local +Z to unit vector v."""
    z = np.asarray(v, float); z = z / np.linalg.norm(z)
    tmp = UP if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(tmp, z); x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return mat2quat(np.column_stack([x, y, z]))


gates = SafetyGates(real_robot=args.real_robot)
# Preliminary angle gate against the nominal board; re-checked after board detection.
gates.check_drill_angles(args.tilt, args.azim, bit_direction(args.tilt, args.azim), BOARD_N_HINT)


def log(msg):
    print(f"[{sim_t():6.2f}s] {msg}", flush=True)


# ------------------------------------------------------------------ build world
rng = np.random.default_rng(args.random) if args.random is not None else None
xy = np.array(args.xy) + (rng.uniform(-0.04, 0.04, 2) if rng is not None else 0)
yaw = np.deg2rad(args.yaw + (rng.uniform(-25, 25) if rng is not None else 0))
model, _ = scene.build_model(drill_xy=tuple(xy), drill_yaw=yaw)
data = mujoco.MjData(model)
data.qpos[:7] = REST_Q
mujoco.mj_forward(model, data)
sim_t = lambda: data.time

robot = Robot(model, data)
robot.q_cmd = REST_Q.copy()
views = ViewPlanner(model)
cam = D435i(model, data, noise=not args.no_noise, seed=args.random or 0)
det = DrillDetector(cam)
board_det = BoardDetector(cam, hole_offset_yz=(0.02, -0.01))
hand = robot.hand
hand.set_angles(hand.OPEN); hand.cmd[:] = hand.target

# ------------------------------------------------------------------ drill physics add-ons
st = dict(motor=False, depth=0.0, phase="start", overlay=None, overlay_until=-1, warned=False,
          holding=False,
          board_n=BOARD_N_HINT.copy(), hole=HOLE_HINT.copy(), drill_dir=bit_direction(args.tilt, args.azim))
trig_q = model.jnt_qposadr[model.joint("trigger").id]
motor_act = model.actuator("drill_motor").id
tip_site = model.site("bit_tip").id
hole_gid = model.geom("hole").id


def drill_logic():
    trig = data.qpos[trig_q]
    if not st["motor"] and trig > TRIGGER_ON:
        st["motor"] = True; log(f"trigger pulled ({trig*1000:.1f} mm) -> motor ON")
    elif st["motor"] and trig < TRIGGER_OFF:
        st["motor"] = False; log("trigger released -> motor OFF")
    data.ctrl[motor_act] = MOTOR_SPEED if st["motor"] else 0.0

    tip = data.site_xpos[tip_site]
    n = st["board_n"]; d = st["drill_dir"]; hole = st["hole"]
    pen = float((tip - hole) @ n)
    dn = float(d @ n)
    if abs(dn) < 1e-6:
        return
    hit = tip - (pen / dn) * d
    rel = hit - scene.BOARD_POS
    inside = abs(rel[1]) < scene.BOARD_HALF[1] and abs(rel[2]) < scene.BOARD_HALF[2]
    max_pen = 2 * scene.BOARD_HALF[0] / max(abs(dn), 0.2) + 0.005
    if inside and 0 < pen < max_pen:
        if st["motor"]:
            if pen > st["depth"]:
                st["depth"] = pen
                model.geom_pos[hole_gid] = [-scene.BOARD_HALF[0] - 0.0005, rel[1], rel[2]]
                model.geom_quat[hole_gid] = quat_align_z(d)
                model.geom_size[hole_gid][:2] = [0.0045, 0.0006]
                model.geom_rgba[hole_gid][3] = 1.0
        elif pen > 0.002 and not st["warned"]:
            st["warned"] = True; log("WARNING: bit pressed into the board without spinning")


robot.callbacks.append(drill_logic)

_board_bid = model.body("board").id
_chuck_bid = model.body("chuck").id


def board_collision_guard():
    """Abort as soon as anything except the chuck/bit touches the board (held motions)."""
    if not st["holding"]:
        return
    for i in range(data.ncon):
        c = data.contact[i]
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        if _board_bid not in (b1, b2):
            continue
        other = b2 if b1 == _board_bid else b1
        if other == _chuck_bid:
            continue
        raise GateError(f"collision: {model.body(other).name} hit the board ({st['phase']})")


def tip_transit_guard():
    """Chuck-board contact is excluded, so the bit can ghost through the plywood.
    After a safe post-grasp retreat, refuse non-drill motions that put the tip into the face."""
    if not st["holding"]:
        return
    ph = str(st["phase"])
    if ph.startswith(("7/", "FAILED")):
        return
    gates.check_transit_tip(
        data.site_xpos[tip_site], st["hole"], st["board_n"], st["phase"],
        board_pos=scene.BOARD_POS, board_half=scene.BOARD_HALF)


def relocate_held_drill(p_goal, R_goal, T=2.5, name="relocate"):
    """Move a grasped drill without sweeping the bit through the board.

    Joint-space shortcuts interpolate a ~20 cm bit on an arc. Reorient only once
    the grasp site is far enough in front of the face, then translate in.
    """
    n = st["board_n"]
    hole = st["hole"]
    p0, R0 = robot.site_pose()
    g = gates.grasp_face_clearance(p0, hole, n)
    need = gates.REORIENT_GRASP_STANDOFF
    p_via = p0.copy()
    if g < need:
        p_via = p0 - (need - g) * n
    p_via[2] = max(float(p_via[2]), float(p0[2]), float(p_goal[2]), 0.18)
    gates.check_workspace(p_via, f"{name}-via")
    if np.linalg.norm(p_via - p0) > 0.015:
        log(f"{name}: pull back {np.linalg.norm(p_via - p0)*1000:.0f} mm, then reorient")
        robot.move_to_pose(p_via, R0, T=1.5)
    robot.move_to_pose(p_via, R_goal, T=1.8)
    robot.move_to_pose(p_goal, R_goal, T=T)


robot.callbacks.append(board_collision_guard)
robot.callbacks.append(tip_transit_guard)
if os.environ.get("DBG_CONTACT"):
    _seen = set()
    def _contact_dbg():
        did = model.body("drill").id
        for c in data.contact[:data.ncon]:
            b1, b2 = model.body_rootid[model.geom_bodyid[c.geom1]], model.body_rootid[model.geom_bodyid[c.geom2]]
            if did in (b1, b2) and 0 not in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]):
                other = model.body(model.geom_bodyid[c.geom1 if b2 == did else c.geom2]).name
                if other not in _seen:
                    _seen.add(other); log(f"first contact drill <-> {other} ({st['phase']})")
    robot.callbacks.append(_contact_dbg)

# ------------------------------------------------------------------ visualisation
recorder = None
if args.video:
    class Recorder:
        def __init__(self, path):
            self.W, self.H = args.video_size
            self.IW, self.IH = self.W // 3, self.H // 3
            self.FPS = args.fps
            self.r = mujoco.Renderer(model, self.H, self.W)
            self.ri = mujoco.Renderer(model, self.IH, self.IW)
            self.vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self.FPS, (self.W, self.H))
            self.cam = mujoco.MjvCamera()
            self.cam.lookat[:] = [0.40, -0.10, 0.26]; self.cam.distance = 1.35
            self.cam.azimuth = 62; self.cam.elevation = -24
            self.every = int(round(1.0 / self.FPS / robot.ctrl_dt)); self.k = 0

        def __call__(self):
            self.k += 1
            if self.k % self.every:
                return
            self.r.update_scene(data, self.cam)
            img = self.r.render()[:, :, ::-1].copy()
            if st["overlay"] is not None and sim_t() < st["overlay_until"]:
                ins = cv2.resize(st["overlay"], (self.IW, self.IH))[:, :, ::-1]
            else:
                self.ri.update_scene(data, camera="d435i_color")
                ins = self.ri.render()[:, :, ::-1]
            img[8:8 + self.IH, self.W - self.IW - 8:self.W - 8] = ins
            cv2.rectangle(img, (self.W - self.IW - 9, 7), (self.W - 7, 9 + self.IH), (255, 255, 255), 1)
            cv2.putText(img, "wrist D435i", (self.W - self.IW - 2, self.IH + 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
            txt = f"{st['phase']}   t={sim_t():5.1f}s   motor={'ON ' if st['motor'] else 'off'}   hole={st['depth']*1000:4.1f} mm"
            cv2.rectangle(img, (0, self.H - 30), (self.W, self.H), (0, 0, 0), -1)
            cv2.putText(img, txt, (10, self.H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            self.vw.write(img)

        def close(self):
            self.vw.release()

    recorder = Recorder(args.video)
    robot.callbacks.append(recorder)

viewer = None
if args.viewer:
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    t_wall0 = [time.time(), data.time]

    def sync():
        viewer.sync()
        lag = (data.time - t_wall0[1]) - (time.time() - t_wall0[0])
        if lag > 0:
            time.sleep(lag)
        if not viewer.is_running():
            raise SystemExit

    robot.callbacks.append(sync)

# live OpenCV window of the palm RealSense (on with --viewer unless --no-wrist-cam)
if args.viewer and not args.no_wrist_cam:
    class WristPreview:
        def __init__(self):
            self.W, self.H = 640, 360
            self.r = mujoco.Renderer(model, self.H, self.W)
            self.every = max(1, int(round(1.0 / 12.0 / robot.ctrl_dt)))
            self.k = 0
            cv2.namedWindow("wrist D435i", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("wrist D435i", self.W, self.H)

        def __call__(self):
            self.k += 1
            if self.k % self.every:
                return
            if st["overlay"] is not None and sim_t() < st["overlay_until"]:
                img = cv2.resize(st["overlay"], (self.W, self.H))
            else:
                self.r.update_scene(data, camera="d435i_color")
                img = self.r.render()
            # MuJoCo returns a non-contiguous RGB view; OpenCV needs contiguous BGR uint8
            img = np.ascontiguousarray(img[:, :, ::-1], dtype=np.uint8)
            cv2.putText(img, str(st["phase"]), (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("wrist D435i", img)
            cv2.waitKey(1)

    robot.callbacks.append(WristPreview())


def phase(name):
    st["phase"] = name; log(f"== {name}")


def show_detection(img, secs=2.5):
    st["overlay"], st["overlay_until"] = img, sim_t() + secs


# ------------------------------------------------------------------ the task
def main():
    gt_grip = data.body("drill").xpos[:2].copy()
    log(f"drill placed at ({xy[0]:.3f}, {xy[1]:.3f}), yaw {np.degrees(yaw):+.1f} deg (ground truth, for reporting only)")
    robot.wait(0.3)

    # 1) coarse look from above --------------------------------------------------
    phase("1/8 look: top-down scan")
    ws = np.array([0.48, 0.0, 0.0])                  # centre of the pick area
    # wide (69 deg) image axis along the table depth so the whole drill incl. bit is in view
    p, R = views.site_target_for_view(ws + [-0.05, 0, 0.60], ws, up_hint=(0, 1, 0))
    robot.goto_pose_joint_space(p, R, T=2.5); robot.wait(0.4)
    rgb, dep = cam.capture()
    c_xy, ax = det.detect_coarse(rgb, dep, cam.pose())
    log(f"coarse: drill near ({c_xy[0]:.3f}, {c_xy[1]:.3f}), axis yaw {np.degrees(np.arctan2(ax[1], ax[0])):+.1f} deg")
    show_detection(rgb, 1.5)

    # 2) side view perpendicular to the drill axis -------------------------------
    phase("2/8 look: side view")
    side = np.cross(UP, [ax[0], ax[1], 0.0])          # the side the hand will come from
    look = np.array([c_xy[0], c_xy[1], 0.11])
    targets = [look + [0, 0, dz] for dz in (-0.03, 0.0, 0.04, 0.09)] + [look + 0.12 * np.array([ax[0], ax[1], 0.8])]
    q = None
    ax3 = np.array([ax[0], ax[1], 0.0])
    # the roll of the camera about its optical axis is free for perception -> try several
    for dist, hgt in [(0.34, 0.25), (0.30, 0.22), (0.38, 0.28)]:
        for up in (UP, -UP, ax3, -ax3):
            p, R = views.site_target_for_view(look + dist * side + [0, 0, hgt], look, up_hint=up)
            for cand in robot.ik_solutions(p, R):
                if views.line_of_sight_clear(model, cand, targets, robot.ik.qadr):
                    q = cand; break
            if q is not None:
                break
        if q is not None:
            break
    if q is None:
        raise RuntimeError("no reachable, unoccluded side view")
    robot.move_joints(q, T=2.5); robot.wait(0.4)
    rgb, dep = cam.capture()
    cv2.imwrite(os.path.join(HERE, "outputs", "side_view_raw.png"), rgb[:, :, ::-1])
    log(f"side view from ({cam.pose()[0][0]:.2f}, {cam.pose()[0][1]:.2f}, {cam.pose()[0][2]:.2f})")
    est = det.detect(rgb, dep, cam.pose(), axis_hint=ax)
    gates.check_perception(est, axis_hint=ax)
    show_detection(det.draw(rgb, est, cam.pose()))
    log(f"drill: grip ({est.grip_xy[0]:.4f}, {est.grip_xy[1]:.4f}) r={est.grip_radius*1000:.1f} mm, "
        f"yaw {np.degrees(est.yaw):+.1f} deg, housing underside z={est.housing_underside_z:.3f}")
    log(f"   perception error vs ground truth: grip {np.linalg.norm(est.grip_xy-gt_grip)*1000:.1f} mm, "
        f"yaw {np.degrees(np.angle(np.exp(1j*(est.yaw-yaw)))):+.1f} deg")

    # 3) grasp plan ---------------------------------------------------------------
    a = est.axis / np.linalg.norm(est.axis)
    bz = np.cross(a, UP)                               # hand finger direction
    Rg = np.column_stack([a, UP, bz])                  # palm normal -> chuck, index side up
    pg = np.array([*est.grip_xy, est.housing_underside_z - GRIP_BELOW_HOUSING])
    tip_in_hand = Rg.T @ (est.bit_tip - pg)            # where the bit will be, in grasp frame

    phase("3/8 approach (sideways, thumb pre-opposed)")
    hand.set_angles(hand.PRE_GRASP[:5] + [0], speed=800)
    # entry point further out and higher, then diagonally in/down so the fingertips stay clear
    # of the housing footprint; prefer a long slide, fall back to a shorter one if unreachable
    # (slide, entry offset outward, entry height); lower entries keep >=35 mm fingertip
    # clearance from the housing axis by the time the fingers reach housing height
    options = [(SLIDE, 0.06, 0.10), (0.12, 0.04, 0.08), (0.12, 0.06, 0.05), (0.12, 0.01, 0.03), (0.12, 0.0, 0.02)]
    for slide, out, up in options:
        p_slide = pg - slide * bz - STANDOFF * a
        p_entry = p_slide - out * bz + [0, 0, up]
        wps = [p_entry, p_slide] + [pg - STANDOFF * a - s * slide * bz for s in (0.66, 0.33, 0.0)] + [pg]
        try:
            q_app, margin = robot.plan_path_ik(wps, Rg)
            break
        except RuntimeError as ex:
            log(f"approach option {slide, out, up} rejected: {ex}")
            continue
    else:
        raise RuntimeError("drill is outside the reachable grasping workspace")
    log(f"approach: slide {slide*100:.0f} cm, joint-limit margin {margin:.2f} rad")
    gates.check_joint_margin(margin)
    gates.check_workspace(pg, "grasp")
    robot.move_joints(q_app, T=3.0)
    robot.move_to_pose(p_slide, Rg, T=1.4)
    e = robot.move_to_pose(pg - STANDOFF * a, Rg, T=2.0)
    log(f"slide max tracking err {robot.max_track_err*1000:.1f} mm; residual {np.round(e[:3]*1000,1)} mm, {np.round(np.degrees(e[3:]),1)} deg; q={np.round(data.qpos[:7],2)}")
    gates.check_tracking(robot.max_track_err, "slide")
    robot.move_to_pose(pg, Rg, T=0.6)

    # 4) grasp + lift + check ------------------------------------------------------
    phase("4/8 grasp")
    Rdr = data.body("drill").xmat.reshape(3, 3)
    log(f"drill disturbance during approach: {np.linalg.norm(data.body('drill').xpos[:2]-gt_grip)*1000:.1f} mm, "
        f"{np.degrees(np.arctan2(Rdr[1,0], Rdr[0,0]) - yaw):+.1f} deg")
    hand.set_angles([0, 0, 0, 1000, 1000, 0], speed=800); robot.wait(1.0)
    hand.set_angles(HAND_GRASP, speed=800); robot.wait(1.0)
    f = hand.get_forces()
    log(f"finger torques [little ring middle index thumb_bend thumb_rot] = {np.round(f, 2)} N*m")
    gates.check_grasp(f)
    z0 = data.body("drill").xpos[2]
    robot.move_to_pose(pg + [0, 0, 0.15], Rg, T=1.8); robot.wait(0.3)
    lifted = data.body("drill").xpos[2] - z0
    log(f"lifted drill by {lifted*1000:.0f} mm (commanded 150)")
    gates.check_lift(lifted)
    # With yaw≈0 the bit already points at the board and, because chuck↔board contact
    # is excluded, the tip can sit inside the face while the drill is still on the
    # table. Pull the tip out along -bit BEFORE enabling the transit guard.
    p_hold, R_hold = robot.site_pose()
    bit = R_hold[:, 0]
    n = st["board_n"]
    tip_clear = gates.tip_clearance(data.site_xpos[tip_site], st["hole"], n)
    SAFE_TIP = 0.08
    if tip_clear < SAFE_TIP:
        pull = -bit if float(bit @ n) > 0.25 else -n
        dn_pull = float(pull @ n)
        if abs(dn_pull) < 0.2:
            pull, dn_pull = -n, -1.0
        # clearance' = tip_clear - s*(pull·n); want clearance' = SAFE_TIP
        s = (tip_clear - SAFE_TIP) / dn_pull
        p_back = p_hold + s * pull
        p_back[2] = max(float(p_back[2]), float(p_hold[2]))
        # also keep the grasp far enough for later reorient
        g0 = gates.grasp_face_clearance(p_back, st["hole"], n)
        if g0 < gates.REORIENT_GRASP_STANDOFF:
            p_back = p_back - (gates.REORIENT_GRASP_STANDOFF - g0) * n
            p_back[2] = max(float(p_back[2]), float(p_hold[2]))
        gates.check_workspace(p_back, "post-lift tip retreat")
        log(f"tip was {tip_clear*1000:.1f} mm into/near board -> retreat "
            f"{np.linalg.norm(p_back - p_hold)*1000:.0f} mm along bit before reorient")
        robot.move_to_pose(p_back, R_hold, T=2.0)
        tip_clear = gates.tip_clearance(data.site_xpos[tip_site], st["hole"], n)
        log(f"tip clearance after retreat: {tip_clear*1000:.1f} mm")
    st["holding"] = True

    # 5) locate the board with the wrist camera ---------------------------------------
    phase("5/8 locate board")
    look = HOLE_HINT.copy()
    # stand off along the nominal board normal so the face fills the wrist view
    cam_guess = look - 0.38 * BOARD_N_HINT + np.array([0.0, 0.0, 0.08])
    p_view, R_view = views.site_target_for_view(cam_guess, look, up_hint=UP)
    gates.check_workspace(p_view, "board-view")
    relocate_held_drill(p_view, R_view, T=2.5, name="board-view")
    robot.wait(0.4)
    rgb, dep = cam.capture()
    board = board_det.detect(rgb, dep, cam.pose(), normal_hint=BOARD_N_HINT)
    gates.check_board(board)
    show_detection(board_det.draw(rgb, board, cam.pose()))
    err_n = np.degrees(np.arccos(np.clip(board.normal @ BOARD_N_HINT, -1, 1)))
    err_h = np.linalg.norm(board.hole - HOLE_HINT) * 1000
    log(f"board: n=({board.normal[0]:+.3f},{board.normal[1]:+.3f},{board.normal[2]:+.3f}), "
        f"hole=({board.hole[0]:.3f},{board.hole[1]:.3f},{board.hole[2]:.3f}), "
        f"pts={board.n_points}, inliers={board.inlier_ratio:.0%}")
    log(f"   vs nominal: normal err {err_n:.1f} deg, hole err {err_h:.1f} mm")

    board_n = board.normal
    hole = board.hole.copy()
    d = bit_direction(args.tilt, args.azim, board_n)
    Rd = aim_rotation(d, board_n)
    gates.check_drill_angles(args.tilt, args.azim, d, board_n)
    st["board_n"], st["hole"], st["drill_dir"] = board_n, hole, d
    log(f"drill angle tilt={args.tilt:+.1f} deg, azim={args.azim:+.1f} deg, "
        f"dir=({d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f})")

    # 6) distance-based approach (keep tip clear of the face; refuse hand collisions) --
    phase("6/8 approach by tip-to-board distance")
    dn = max(float(d @ board_n), 0.35)
    # far pose already in drill orientation; tip sits APPROACH_CLEAR in front of the face
    p_far = hole - Rd @ tip_in_hand - (APPROACH_CLEAR / dn) * d
    gates.check_workspace(p_far, "approach-far")
    gates.check_aim_clearance(p_far, Rd, tip_in_hand, hole, board_n, along_d=0.0)
    relocate_held_drill(p_far, Rd, T=2.0, name="approach-far")
    robot.wait(0.2)

    def tip_world():
        p, R = robot.site_pose()
        return p + R @ tip_in_hand

    clear = gates.tip_clearance(tip_world(), hole, board_n)
    log(f"approach start: tip-to-face {clear*1000:.1f} mm (target {AIM_STANDOFF*1000:.0f} mm)")
    for k in range(50):
        clear = gates.tip_clearance(tip_world(), hole, board_n)
        err = clear - AIM_STANDOFF
        if abs(err) <= CLEAR_TOL:
            log(f"approach settled: tip-to-face {clear*1000:.1f} mm after {k} corrections")
            break
        # move tip toward/away along the bit; convert face-clearance error into travel on d
        step = float(np.clip(err / dn, -0.035, 0.035))
        p_now, _ = robot.site_pose()
        p_next = p_now + step * d
        gates.check_workspace(p_next, "approach")
        gates.check_aim_clearance(p_next, Rd, tip_in_hand, hole, board_n, along_d=0.0)
        robot.move_to_pose(p_next, Rd, T=max(0.35, abs(step) / 0.04))
        if k % 5 == 0:
            log(f"  correct {k}: clear {clear*1000:.1f} mm -> step {step*1000:+.1f} mm along bit")
    else:
        raise GateError(f"approach failed to reach standoff (clearance {clear*1000:.1f} mm)")

    p_aim, _ = robot.site_pose()
    gates.check_tracking(robot.max_track_err, "approach")
    gates.check_trigger_clearance(tip_world(), hole, board_n)
    feed = AIM_STANDOFF + (DRILL_DEPTH + FEED_MARGIN) / dn
    gates.check_aim_clearance(p_aim, Rd, tip_in_hand, hole, board_n, along_d=0.0)
    gates.check_aim_clearance(p_aim, Rd, tip_in_hand, hole, board_n, along_d=feed)

    # 7) drill -------------------------------------------------------------------------
    phase("7/8 pull trigger + drill")
    hand.set_angles(HAND_TRIGGER, speed=500); robot.wait(1.2)
    if not st["motor"]:
        Rdr = data.body("drill").xmat.reshape(3, 3); pdr = data.body("drill").xpos
        log(f"diag: index tip in drill frame {np.round(Rdr.T @ (data.site_xpos[model.site('index_tip').id] - pdr), 4)}, "
            f"trigger at {model.body('trigger').pos}, grasp site in drill frame "
            f"{np.round(Rdr.T @ (data.site_xpos[robot.ik.site] - pdr), 4)}")
        raise RuntimeError(f"trigger not pressed enough ({data.qpos[trig_q]*1000:.2f} mm)")
    T_feed = gates.feed_duration(feed, 5.0)
    robot.move_to_pose(p_aim + feed * d, Rd, T=T_feed, settle=0.5)
    log(f"hole depth reached {st['depth']*1000:.1f} mm (along board normal)")
    robot.move_to_pose(p_aim, Rd, T=2.0)
    hand.set_angles(HAND_GRASP, speed=500); robot.wait(1.0)
    robot.move_to_pose(p_aim - 0.10 * d + 0.06 * UP, Rd, T=1.5)   # back off along -d

    # 8) put it back -----------------------------------------------------------------
    if not args.no_place:
        phase("8/8 place the drill back")
        relocate_held_drill(pg + np.array([0.0, 0.0, 0.12]), Rg, T=2.5, name="return")
        p_place = pg + [0, 0, 0.004]
        wps = [pg + [0, 0, 0.12], pg + [0, 0, 0.06], p_place, pg - STANDOFF * a, p_slide]
        q_pl, margin = robot.plan_path_ik(wps, Rg, min_margin=0.1)
        robot.move_joints(q_pl, T=3.0)
        robot.move_to_pose(p_place, Rg, T=2.0); robot.wait(0.3)
        log(f"place: max tracking err {robot.max_track_err*1000:.1f} mm")
        def tilt():
            return np.degrees(np.arccos(np.clip(data.body("drill").xmat.reshape(3, 3)[2, 2], -1, 1)))
        hand.set_angles([1000, 1000, 1000, 1000, 1000, 0], speed=600); robot.wait(1.2)
        log(f"  released, tilt {tilt():.1f}")
        robot.move_to_pose(pg - STANDOFF * a, Rg, T=0.6)
        robot.move_to_pose(p_slide, Rg, T=1.8)
        log(f"  slid out, tilt {tilt():.1f}, max tracking {robot.max_track_err*1000:.1f} mm")
        # back away along -axis (toward the robot) until the fingers are behind the housing, then up
        robot.move_to_pose(p_slide - 0.09 * a, Rg, T=1.2)
        robot.move_to_pose(p_slide - 0.09 * a + [0, 0, 0.12], Rg, T=1.2)
        log(f"  backed off, tilt {tilt():.1f}, max tracking {robot.max_track_err*1000:.1f} mm")
        hand.set_angles(hand.OPEN, speed=800)
        robot.move_joints(REST_Q, T=2.5)
        log(f"  home, tilt {tilt():.1f}")
        up_axis = data.body("drill").xmat.reshape(3, 3)[:, 2]
        log(f"drill back on the table, tilt {np.degrees(np.arccos(np.clip(up_axis[2], -1, 1))):.1f} deg, "
            f"displaced {np.linalg.norm(data.body('drill').xpos[:2]-gt_grip)*1000:.0f} mm from start")
    phase("done")
    robot.wait(1.0)


if __name__ == "__main__":
    ok = True
    try:
        main()
    except (RuntimeError, GateError) as e:
        ok = False
        st["holding"] = False
        st["phase"] = f"FAILED: {e}"
        log(f"FAILED: {e}")
        robot.wait(1.0)
    finally:
        if recorder:
            recorder.close(); print(f"video written to {args.video}")
        os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
        cv2.imwrite(os.path.join(HERE, "outputs", "last_detection.png"),
                    st["overlay"][:, :, ::-1] if st["overlay"] is not None else np.zeros((10, 10, 3)))
        cv2.destroyAllWindows()
    if viewer is not None:
        while viewer.is_running():
            time.sleep(0.05); viewer.sync()
    sys.exit(0 if ok else 1)
