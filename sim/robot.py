"""
Robot interface for xArm7 + Inspire RH56E2 in MuJoCo.

 * ArmIK           : damped-least-squares IK on any site, with joint-limit clamping
                     and a nullspace pull toward a comfortable posture.
 * Robot.servo_to  : closed-loop Cartesian servo (integrates task error into the
                     position-actuator setpoints -> zero steady-state error under load).
 * InspireHand     : mirrors the RH56 serial API: 6 values in [0,1000],
                     order = [little, ring, middle, index, thumb_bend, thumb_rot],
                     1000 = open / extended, 0 = closed.
"""
import numpy as np
import mujoco
from calibration import camera_pose_from_tcp
from hw.defaults import REST_Q

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_ACTS = [f"act{i}" for i in range(1, 8)]
HAND_ACTS = ["little", "ring", "middle", "index", "thumb_bend", "thumb_rot"]


# ---------------------------------------------------------------- math helpers
def mat2quat(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9)); return q


def quat2mat(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, q); return R.reshape(3, 3)


def orientation_error(R_target, R_current):
    """Rotation vector (world frame) taking R_current to R_target."""
    qt, qc = mat2quat(R_target), mat2quat(R_current)
    dq = np.zeros(3); mujoco.mju_subQuat(dq, qt, qc)   # in current body frame
    return R_current @ dq


def slerp(q0, q1, s):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if np.dot(q0, q1) < 0: q1 = -q1
    d = np.clip(np.dot(q0, q1), -1, 1)
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th)
    return q / np.linalg.norm(q)


def min_jerk(s):
    s = np.clip(s, 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


# ---------------------------------------------------------------- IK
class ArmIK:
    def __init__(self, model, site="grasp_site"):
        self.m = model
        self.d = mujoco.MjData(model)          # scratch data for offline solves
        self.site = model.site(site).id
        self.qadr = np.array([model.jnt_qposadr[model.joint(j).id] for j in ARM_JOINTS])
        self.dadr = np.array([model.jnt_dofadr[model.joint(j).id] for j in ARM_JOINTS])
        self.lo = model.jnt_range[[model.joint(j).id for j in ARM_JOINTS], 0]
        self.hi = model.jnt_range[[model.joint(j).id for j in ARM_JOINTS], 1]
        self.jp = np.zeros((3, model.nv)); self.jr = np.zeros((3, model.nv))

    def task_error(self, d, p_t, R_t):
        p = d.site_xpos[self.site]; R = d.site_xmat[self.site].reshape(3, 3)
        return np.concatenate([p_t - p, orientation_error(R_t, R)])

    def dq(self, d, p_t, R_t, damping=0.03, rot_weight=0.6, null_gain=0.05, rest=REST_Q):
        mujoco.mj_jacSite(self.m, d, self.jp, self.jr, self.site)
        J = np.vstack([self.jp[:, self.dadr], rot_weight * self.jr[:, self.dadr]])
        e = self.task_error(d, p_t, R_t); e[3:] *= rot_weight
        JJt = J @ J.T + damping**2 * np.eye(6)
        Jpinv = J.T @ np.linalg.inv(JJt)
        q = d.qpos[self.qadr]
        null = (np.eye(7) - Jpinv @ J) @ (null_gain * (rest - q))
        return Jpinv @ e + null, e

    def solve(self, p_t, R_t, q_seed, iters=400, tol=(1e-4, 1e-3)):
        """Offline IK (kinematics only). Returns (q, success, err)."""
        d = self.d
        d.qpos[:] = 0; d.qpos[self.qadr] = q_seed
        for _ in range(iters):
            mujoco.mj_kinematics(self.m, d); mujoco.mj_comPos(self.m, d)
            step, e = self.dq(d, p_t, R_t, null_gain=0.02)
            if np.linalg.norm(e[:3]) < tol[0] and np.linalg.norm(e[3:]) < tol[1]:
                return d.qpos[self.qadr].copy(), True, e
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + np.clip(step, -0.2, 0.2), self.lo, self.hi)
        mujoco.mj_kinematics(self.m, d)
        e = self.task_error(d, p_t, R_t)
        return d.qpos[self.qadr].copy(), bool(np.linalg.norm(e[:3]) < 5e-3 and np.linalg.norm(e[3:]) < 3e-2), e


# ---------------------------------------------------------------- hand
class InspireHand:
    """RH56-style API. angles: list of 6 ints in [0,1000] (1000 = open)."""

    def __init__(self, model, data):
        self.m, self.d = model, data
        self.aid = np.array([model.actuator(a).id for a in HAND_ACTS])
        self.hi = model.actuator_ctrlrange[self.aid, 1]
        self.target = np.full(6, 1000.0)
        self.cmd = self.target.copy()
        self.speed = 1000.0      # units/s (real RH56: speedSet 0..1000)
        self.force_scale = 1.0   # calib.hand.force_scale; 1.0 = sim actuator N*m

    def set_angles(self, angles, speed=None):
        self.target = np.clip(np.asarray(angles, float), 0, 1000)
        if speed is not None: self.speed = speed

    def step(self, dt):
        """Rate-limit toward target (the real hand moves at a fixed commanded speed)."""
        dmax = self.speed * dt * 2.0
        self.cmd += np.clip(self.target - self.cmd, -dmax, dmax)
        self.d.ctrl[self.aid] = (1.0 - self.cmd / 1000.0) * self.hi

    def get_angles(self):
        q = np.array([self.d.qpos[self.m.jnt_qposadr[self.m.actuator_trnid[a, 0]]] for a in self.aid])
        return np.round(1000 * (1 - q / self.hi)).astype(int)

    def get_forces(self):
        """Finger load in the same units as `gates.finger_load` (sim: N*m)."""
        return self.d.actuator_force[self.aid].copy() * float(self.force_scale)

    # handy presets
    OPEN = [1000, 1000, 1000, 1000, 1000, 1000]
    PRE_GRASP = [1000, 1000, 1000, 1000, 1000, 250]     # thumb swung into opposition, extended


# ---------------------------------------------------------------- robot
class Robot:
    def __init__(self, model, data, site="grasp_site", ctrl_dt=0.01):
        self.m, self.d = model, data
        self.ik = ArmIK(model, site)
        self.hand = InspireHand(model, data)
        self.aact = np.array([model.actuator(a).id for a in ARM_ACTS])
        self.q_cmd = data.qpos[self.ik.qadr].copy()
        self.ctrl_dt = ctrl_dt
        self.nsub = max(1, int(round(ctrl_dt / model.opt.timestep)))
        self.windup = 0.12       # rad, max setpoint-vs-measured offset during servoing
        self.callbacks = []      # called every control tick (rendering, logging, drill logic)
        self.rest_q = REST_Q.copy()
        self.max_track_err = 0.0

    @property
    def qadr(self):
        """Joint qpos indices — task uses this instead of `robot.ik.qadr`."""
        return self.ik.qadr

    def get_q(self):
        """Measured arm joints (controller feedback analogue)."""
        return self.d.qpos[self.ik.qadr].copy()

    # -- low level
    def tick(self):
        self.hand.step(self.ctrl_dt)
        self.d.ctrl[self.aact] = self.q_cmd
        for _ in range(self.nsub):
            mujoco.mj_step(self.m, self.d)
        for cb in self.callbacks:
            cb()

    def site_pose(self):
        s = self.ik.site
        return self.d.site_xpos[s].copy(), self.d.site_xmat[s].reshape(3, 3).copy()

    # -- motions
    def wait(self, t):
        for _ in range(int(t / self.ctrl_dt)):
            self.tick()

    def move_joints(self, q_goal, T=2.0):
        q0 = self.q_cmd.copy()
        n = int(T / self.ctrl_dt)
        for k in range(1, n + 1):
            self.q_cmd = q0 + min_jerk(k / n) * (q_goal - q0)
            self.tick()

    def move_to_pose(self, p_goal, R_goal, T=2.0, settle=0.3, gain=0.6):
        """Cartesian min-jerk path + closed-loop servo along it."""
        p0, R0 = self.site_pose()
        q0, q1 = mat2quat(R0), mat2quat(R_goal)
        n = int(T / self.ctrl_dt); ns = int(settle / self.ctrl_dt)
        self.max_track_err = 0.0
        for k in range(1, n + ns + 1):
            s = min_jerk(min(k / n, 1.0))
            p_t = p0 + s * (p_goal - p0)
            R_t = quat2mat(slerp(q0, q1, s))
            self.servo_step(p_t, R_t, gain)
            self.tick()
            self.max_track_err = max(self.max_track_err, np.linalg.norm(self.ik.task_error(self.d, p_t, R_t)[:3]))
        return self.ik.task_error(self.d, p_goal, R_goal)

    def servo_step(self, p_t, R_t, gain=0.6, max_step=0.03):
        # no nullspace pull while servoing: posture is chosen by the offline IK plan
        step, _ = self.ik.dq(self.d, p_t, R_t, null_gain=0.0)
        q = self.d.qpos[self.ik.qadr]
        q_new = self.q_cmd + np.clip(gain * step, -max_step, max_step)
        # anti-windup: never let the setpoint run away from the measured joints
        q_new = q + np.clip(q_new - q, -self.windup, self.windup)
        self.q_cmd = np.clip(q_new, self.ik.lo, self.ik.hi)

    def ik_solutions(self, p_goal, R_goal, n_random=16):
        """All distinct IK solutions from many seeds, sorted by distance to the current posture."""
        cur = self.d.qpos[self.ik.qadr].copy()
        rng = np.random.default_rng(1)
        seeds = [cur, REST_Q] + [np.clip(REST_Q + rng.normal(0, 0.7, 7), self.ik.lo, self.ik.hi)
                                 for _ in range(n_random)]
        sols = []
        for sd in seeds:
            q, ok, _ = self.ik.solve(p_goal, R_goal, sd)
            if ok and all(np.linalg.norm(q - s) > 0.1 for s in sols):
                sols.append(q)
        return sorted(sols, key=lambda q: np.linalg.norm(q - cur))

    def limit_margin(self, q):
        """Smallest distance (rad) to a joint limit, over the joints with real limits."""
        lim = (self.ik.hi - self.ik.lo) < 6.0
        return float(np.min(np.minimum(q - self.ik.lo, self.ik.hi - q)[lim]))

    def plan_path_ik(self, waypoints, R, min_margin=0.2, max_jump=1.0):
        """IK branch for a straight Cartesian path: solve the first waypoint from many seeds,
        then chain the rest; keep the chain with the best joint-limit margin."""
        best = None
        for q0 in self.ik_solutions(waypoints[0], R, n_random=20):
            chain, q, ok = [q0], q0, True
            for w in waypoints[1:]:
                qn, okn, _ = self.ik.solve(w, R, q)
                if not okn or np.max(np.abs(qn - q)) > max_jump:
                    ok = False; break
                chain.append(qn); q = qn
            if not ok:
                continue
            m = min(self.limit_margin(c) for c in chain)
            if best is None or m > best[0]:
                best = (m, chain)
        if best is None:
            raise RuntimeError("no continuous IK chain along the approach path")
        if best[0] < min_margin:
            raise RuntimeError(f"best joint-limit margin only {best[0]:.2f} rad")
        return best[1], best[0]

    def move_joint_path(self, chain, T_first=3.0, T_step=0.8):
        """Follow a solved IK chain in joint space (stays on that branch)."""
        if not chain:
            return
        self.move_joints(chain[0], T=T_first)
        for q in chain[1:]:
            self.move_joints(q, T=T_step)

    def plan_ik(self, p_goal, R_goal, seed=None, n_random=12):
        """Offline IK from several seeds; prefers the solution closest to the current posture."""
        cur = self.d.qpos[self.ik.qadr].copy()
        seeds = [cur if seed is None else seed, REST_Q]
        rng = np.random.default_rng(0)
        seeds += [np.clip(REST_Q + rng.normal(0, 0.6, 7), self.ik.lo, self.ik.hi) for _ in range(n_random)]
        best = None
        for sd in seeds:
            q, ok, e = self.ik.solve(p_goal, R_goal, sd)
            if ok:
                cost = np.linalg.norm(q - cur)
                if best is None or cost < best[0]:
                    best = (cost, q, e)
                if cost < 1.0:
                    break
        if best is None:
            return q, False, e
        return best[1], True, best[2]

    def goto_pose_joint_space(self, p_goal, R_goal, T=2.5, seed=None):
        """Plan with offline IK, then move in joint space (good for large reorientations)."""
        seed = self.d.qpos[self.ik.qadr] if seed is None else seed
        q, ok, e = self.plan_ik(p_goal, R_goal, seed)
        if not ok:
            raise RuntimeError(f"IK failed, residual {np.round(e, 4)}")
        self.move_joints(q, T)
        return q


# ---------------------------------------------------------------- camera viewpoint helper
def look_at_rotation(forward, up_hint):
    """Hand ('rh_base') rotation so the wrist camera looks along `forward` with image-up ~ `up_hint`.
    Camera frame: x=right, y=up, z=back; mounted with -z_cam = +x_hand, y_cam = y_hand, x_cam = z_hand."""
    f = np.asarray(forward, float); f /= np.linalg.norm(f)
    x_cam = np.cross(f, up_hint); x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(-f, x_cam)          # y = z_cam x x_cam  (z_cam = -f)
    R = np.column_stack([f, y_cam, x_cam])
    assert np.linalg.det(R) > 0.99
    return np.column_stack([f, y_cam, x_cam])


class ViewPlanner:
    """Maps a desired camera pose to a grasp_site target via T_ee_cam (not cam_xpos)."""

    def __init__(self, hand_eye, model=None, cam="d435i_color", site="grasp_site"):
        self.hand_eye = hand_eye
        self.model = model
        self.cam_name = cam
        self.site_name = site

    def line_of_sight_clear(self, model, q, targets, arm_qadr, robot_root="link_base", cam=None):
        """Ray-cast from the calibrated camera (at arm config q) to each target.

        On hardware there is no MuJoCo model — skip the occlusion check.
        """
        if model is None:
            return True
        cam = self.cam_name if cam is None else cam
        d = mujoco.MjData(model)
        d.qpos[:] = model.qpos0; d.qpos[arm_qadr] = q
        mujoco.mj_forward(model, d)
        root = model.body(robot_root).id
        robot_bodies = {b for b in range(model.nbody) if self._is_desc(model, b, root)}
        p_ee = d.site_xpos[model.site(self.site_name).id]
        R_ee = d.site_xmat[model.site(self.site_name).id].reshape(3, 3)
        c, _ = camera_pose_from_tcp(p_ee, R_ee, self.hand_eye)
        gid = np.zeros(1, np.int32)
        for t in targets:
            v = np.asarray(t) - c; L = np.linalg.norm(v)
            dist = mujoco.mj_ray(model, d, c, v / L, None, 1, model.body(model.cam_bodyid[model.camera(cam).id]).id, gid)
            if dist >= 0 and dist < L - 0.03 and model.geom_bodyid[gid[0]] in robot_bodies:
                return False
        return True

    @staticmethod
    def _is_desc(model, b, root):
        while b > 0:
            if b == root:
                return True
            b = model.body_parentid[b]
        return False

    def site_target_for_view(self, cam_pos, look_at, up_hint=(0, 0, 1)):
        R = look_at_rotation(np.asarray(look_at) - np.asarray(cam_pos), np.asarray(up_hint, float))
        p_ee = np.asarray(cam_pos, float) - R @ np.asarray(self.hand_eye.t, float)
        return p_ee, R
