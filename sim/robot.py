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
from control import CartesianPI, duration_for_distance, min_jerk, polyline_from_qs, q_along_polyline
from hw.defaults import REST_Q
from hw.ik_select import branch_cost, limit_margin as _limit_margin, rank_qs

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
        return d.qpos[self.qadr].copy(), bool(np.linalg.norm(e[:3]) < 1e-3 and np.linalg.norm(e[3:]) < 1.5e-2), e


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
        dmax = self.speed * dt
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
        self.vmax_cart = 0.12    # m/s default cap for unspecified Cartesian hops
        self.pi = CartesianPI(ctrl_dt)
        self.last_saturated = False
        self.last_cmd_lag = 0.0
        self.last_tcp_err = 0.0
        self._plan_d = None
        self._geom_groups = None

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

    def fk_site(self, q):
        """TCP pose at arm joints `q` (scratch FK; does not move the live robot)."""
        d = self.ik.d
        d.qpos[:] = self.d.qpos
        d.qpos[self.ik.qadr] = np.asarray(q, float)
        mujoco.mj_kinematics(self.m, d)
        s = self.ik.site
        return d.site_xpos[s].copy(), d.site_xmat[s].reshape(3, 3).copy()

    def contact_pairs(self, limit=8):
        """Live contact geom-name pairs (`a|b`), longest first."""
        d, m = self.d, self.m
        n = int(getattr(d, "ncon", 0))
        if n <= 0:
            return []
        out = []
        for i in range(min(n, int(limit) * 3)):
            c = d.contact[i]
            g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)) or str(int(c.geom1))
            g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)) or str(int(c.geom2))
            out.append(f"{g1}|{g2}")
            if len(out) >= int(limit):
                break
        return out

    def _ensure_plan_d(self):
        """Dedicated MjData for kinematic prediction. Never zero qpos (free joints)."""
        if self._plan_d is None:
            self._plan_d = mujoco.MjData(self.m)
        return self._plan_d

    def _set_plan_q(self, q):
        d = self._ensure_plan_d()
        d.qpos[:] = self.d.qpos
        d.qpos[self.ik.qadr] = np.asarray(q, float)
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        return d

    def _geom_id_groups(self):
        if self._geom_groups is not None:
            return self._geom_groups
        robot, table, drill = [], [], []
        # Distal arm + palm only — full-mesh × drill pairs made planning stall for minutes.
        robot_bodies = ("link4", "link5", "link6", "link7", "rh_base", "hand_base_link")
        table_names = {"table", "board", "board_leg"}
        drill_names = {
            "battery", "handle", "handle_grip", "body", "nose", "chuck",
        }
        for i in range(int(self.m.ngeom)):
            if int(self.m.geom_contype[i]) == 0 and int(self.m.geom_conaffinity[i]) == 0:
                continue
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            bid = int(self.m.geom_bodyid[i])
            bname = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            is_robot = (
                bname in robot_bodies
                or name.startswith(("d435", "end_tool"))
                or "flange" in bname
            )
            if is_robot:
                robot.append(i)
            if name in table_names or bname == "board":
                table.append(i)
            if name in drill_names or bname == "drill":
                drill.append(i)
        self._geom_groups = (
            np.asarray(robot, int), np.asarray(table, int), np.asarray(drill, int),
        )
        return self._geom_groups

    def arm_clearance(self, q, allow_finger_handle=False, distmax=0.08):
        """Min geom distance (m) from distal arm to table/board/(drill). 0 if overlapping."""
        d = self._set_plan_q(q)
        robot_ids, table_ids, drill_ids = self._geom_id_groups()
        obs_ids = table_ids if allow_finger_handle else np.concatenate(
            [table_ids, drill_ids]) if table_ids.size or drill_ids.size else table_ids
        if robot_ids.size == 0 or obs_ids.size == 0:
            return 1.0, ""
        best = float(distmax)
        pair = ""
        fromto = np.zeros(6)
        for g1 in robot_ids:
            for g2 in obs_ids:
                dist = float(mujoco.mj_geomDistance(
                    self.m, d, int(g1), int(g2), float(distmax), fromto))
                if dist < best:
                    best = dist
                    n1 = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, int(g1)) or str(g1)
                    n2 = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, int(g2)) or str(g2)
                    pair = f"{n1}|{n2}"
                if best <= 0.0:
                    return 0.0, pair
        return max(0.0, best), pair

    def predict_cart_servo(self, q_entry, targets, Rg, hops=None, gain=0.6,
                           max_step=0.03, tol=0.020):
        """Kinematic replay of the executor's Cartesian hops from `q_entry`.

        Does not step physics. Returns `(ok, traj, err_m, tag)` where `tag` is
        the failing target index or -1 on success.
        """
        targets = [np.asarray(t, float) for t in targets]
        if not targets:
            return True, [np.asarray(q_entry, float).copy()], 0.0, -1
        hops = list(hops) if hops is not None else [0.03] * len(targets)
        while len(hops) < len(targets):
            hops.append(hops[-1])
        d = self._ensure_plan_d()
        d.qpos[:] = self.d.qpos
        q = np.asarray(q_entry, float).copy()
        d.qpos[self.ik.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        Rg = np.asarray(Rg, float)
        traj = [q.copy()]
        site = self.ik.site
        for ti, (tgt, hop) in enumerate(zip(targets, hops)):
            p0 = d.site_xpos[site].copy()
            dist = float(np.linalg.norm(tgt - p0))
            nseg = max(1, int(np.ceil(dist / max(float(hop), 1e-3))))
            for i in range(1, nseg + 1):
                pi = tgt if i == nseg else p0 + (i / nseg) * (tgt - p0)
                ok_hop = False
                for _ in range(25):
                    step, _ = self.ik.dq(d, pi, Rg, null_gain=0.0)
                    q = np.clip(q + np.clip(gain * step, -max_step, max_step),
                                self.ik.lo, self.ik.hi)
                    d.qpos[self.ik.qadr] = q
                    mujoco.mj_kinematics(self.m, d)
                    mujoco.mj_comPos(self.m, d)
                    traj.append(q.copy())
                    if float(np.linalg.norm(d.site_xpos[site] - pi)) < float(tol):
                        ok_hop = True
                        break
                if not ok_hop:
                    err = float(np.linalg.norm(d.site_xpos[site] - pi))
                    return False, traj, err, ti
        err = float(np.linalg.norm(d.site_xpos[site] - targets[-1]))
        return True, traj, err, -1

    def traj_min_clearance(self, traj, pg=None, subsample_m=0.025, min_clear=0.005):
        """Sweep predicted qs; whitelist finger↔handle in the last 4 cm to grip."""
        pg = None if pg is None else np.asarray(pg, float)
        best, pair = 1.0, ""
        last_p = None
        site = self.ik.site
        for q in traj:
            d = self._set_plan_q(q)
            p = d.site_xpos[site].copy()
            if last_p is not None and float(np.linalg.norm(p - last_p)) < float(subsample_m):
                continue
            last_p = p
            remain = 1.0 if pg is None else float(np.linalg.norm(p - pg))
            c, who = self.arm_clearance(q, allow_finger_handle=remain < 0.12)
            if c < best:
                best, pair = c, who
            if best < float(min_clear):
                return best, pair
        return best, pair

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

    def move_to_pose(self, p_goal, R_goal, T=2.0, settle=0.3, gain=0.6, vmax=None):
        """Cartesian min-jerk path + closed-loop PI servo along it."""
        p0, R0 = self.site_pose()
        p_goal = np.asarray(p_goal, float)
        T = duration_for_distance(np.linalg.norm(p_goal - p0), T,
                                  vmax if vmax is not None else 1e6)
        q0, q1 = mat2quat(R0), mat2quat(R_goal)
        n = max(1, int(T / self.ctrl_dt)); ns = int(settle / self.ctrl_dt)
        self.max_track_err = 0.0
        self.last_saturated = False
        self.last_cmd_lag = 0.0
        self.last_tcp_err = 0.0
        self.pi.reset()
        for k in range(1, n + ns + 1):
            s = min_jerk(min(k / n, 1.0))
            p_t = p0 + s * (p_goal - p0)
            R_t = quat2mat(slerp(q0, q1, s))
            self.servo_step(p_t, R_t, gain)
            self.tick()
            self.max_track_err = max(self.max_track_err, np.linalg.norm(self.ik.task_error(self.d, p_t, R_t)[:3]))
        return self.ik.task_error(self.d, p_goal, R_goal)

    def move_along(self, direction, distance_m, R_hold, T=1.0, speed=None, settle=0.05):
        """TCP hop along a world direction (tool-axis feed / retract). Holds `R_hold`."""
        d = np.asarray(direction, float)
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-12:
            raise ValueError("move_along: direction has zero length")
        d = d / nrm
        dist = float(distance_m)
        vmax = float(speed) if speed is not None else self.vmax_cart
        p0, _ = self.site_pose()
        return self.move_to_pose(p0 + dist * d, R_hold, T=T, settle=settle, vmax=vmax)

    def servo_step(self, p_t, R_t, gain=0.6, max_step=0.03):
        # no nullspace pull while servoing: posture is chosen by the offline IK plan
        p_pi = self.pi.target(p_t)
        step, _ = self.ik.dq(self.d, p_pi, R_t, null_gain=0.0)
        q = self.d.qpos[self.ik.qadr]
        q_new = self.q_cmd + np.clip(gain * step, -max_step, max_step)
        # anti-windup: never let the setpoint run away from the measured joints
        delta = q_new - q
        saturated = bool(np.any(np.abs(delta) >= self.windup - 1e-9))
        q_new = q + np.clip(delta, -self.windup, self.windup)
        self.q_cmd = np.clip(q_new, self.ik.lo, self.ik.hi)
        e = self.ik.task_error(self.d, p_t, R_t)
        self.pi.update(e[:3], saturated=saturated)
        self.last_saturated = bool(self.last_saturated or saturated)
        self.last_cmd_lag = max(self.last_cmd_lag, float(np.max(np.abs(self.q_cmd - q))))
        self.last_tcp_err = float(np.linalg.norm(e[:3]))

    def ik_solutions(self, p_goal, R_goal, n_random=16):
        """Distinct IK solutions, most stable branch first (not just nearest q)."""
        cur = self.d.qpos[self.ik.qadr].copy()
        rng = np.random.default_rng(1)
        seeds = [cur, REST_Q] + [np.clip(REST_Q + rng.normal(0, 0.7, 7), self.ik.lo, self.ik.hi)
                                 for _ in range(n_random)]
        sols = []
        for sd in seeds:
            q, ok, _ = self.ik.solve(p_goal, R_goal, sd)
            if ok and all(np.linalg.norm(q - s) > 0.1 for s in sols):
                sols.append(q)
        return rank_qs(sols, cur, rest=self.rest_q, lo=self.ik.lo, hi=self.ik.hi)

    def limit_margin(self, q):
        """Smallest distance (rad) to a joint limit, over the joints with real limits."""
        return _limit_margin(q, self.ik.lo, self.ik.hi)

    def plan_path_candidates(self, waypoints, R, min_margin=0.2, max_jump=1.0):
        """All IK chains along `waypoints`, lowest branch-cost first. `(cost, chain, margin)`."""
        chains = []
        for q0 in self.ik_solutions(waypoints[0], R, n_random=20):
            chain, q, ok = [q0], q0, True
            for w in waypoints[1:]:
                qn, okn, _ = self.ik.solve(w, R, q)
                if not okn or np.max(np.abs(qn - q)) > max_jump:
                    ok = False; break
                chain.append(qn); q = qn
            if ok:
                chains.append(chain)
        if not chains:
            raise RuntimeError("no continuous IK chain along the approach path")
        ranked = []
        q_cur = self.get_q()
        for chain in chains:
            m = min(self.limit_margin(c) for c in chain)
            if m < min_margin:
                continue
            cost = branch_cost(chain, q_cur, rest=self.rest_q, lo=self.ik.lo, hi=self.ik.hi)
            ranked.append((cost, chain, m))
        if not ranked:
            best_m = max(min(self.limit_margin(c) for c in ch) for ch in chains)
            raise RuntimeError(f"best joint-limit margin only {best_m:.2f} rad")
        ranked.sort(key=lambda t: t[0])
        return ranked

    def plan_path_ik(self, waypoints, R, min_margin=0.2, max_jump=1.0):
        """IK branch along a Cartesian polyline: stable (short, unwrapped) over max-margin."""
        ranked = self.plan_path_candidates(waypoints, R, min_margin=min_margin, max_jump=max_jump)
        _, chain, margin = ranked[0]
        return chain, margin

    def move_joint_path(self, chain, T_first=3.0, T_step=0.8):
        """Follow an IK chain as one min-jerk in joint-space arc length (no via stops)."""
        if not chain:
            return
        qs = [self.q_cmd.copy()] + [np.asarray(q, float).reshape(7) for q in chain]
        segs = polyline_from_qs(qs)
        if not segs:
            return
        total = sum(seg[2] for seg in segs)
        T = max(float(T_first) + float(T_step) * max(len(chain) - 1, 0), total / 1.2)
        n = max(1, int(T / self.ctrl_dt))
        for k in range(1, n + 1):
            self.q_cmd = q_along_polyline(segs, min_jerk(k / n) * total)
            self.tick()

    def plan_ik(self, p_goal, R_goal, seed=None, n_random=12):
        """Offline IK from several seeds; prefers the stable branch, not only nearest q."""
        cur = self.d.qpos[self.ik.qadr].copy()
        seeds = [cur if seed is None else seed, REST_Q]
        rng = np.random.default_rng(0)
        seeds += [np.clip(REST_Q + rng.normal(0, 0.6, 7), self.ik.lo, self.ik.hi) for _ in range(n_random)]
        best = None
        last_e = None
        q = None
        for sd in seeds:
            q, ok, e = self.ik.solve(p_goal, R_goal, sd)
            last_e = e
            if not ok:
                continue
            cost = branch_cost([q], cur, rest=self.rest_q, lo=self.ik.lo, hi=self.ik.hi)
            if best is None or cost < best[0]:
                best = (cost, q, e)
        if best is None:
            return q, False, last_e
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
    up = np.asarray(up_hint, float)
    x_cam = np.cross(f, up)
    if np.linalg.norm(x_cam) < 1e-8:
        alt = np.array([0.0, 1.0, 0.0]) if abs(f[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x_cam = np.cross(f, alt)
    x_cam /= np.linalg.norm(x_cam)
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
