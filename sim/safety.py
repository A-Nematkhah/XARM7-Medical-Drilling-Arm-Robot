"""
Safety gates between the task planner and the robot (sim today, hardware later).

Every motion that would be dangerous on a real xArm7 + RH56 + spinning drill
must pass these checks. Keep limits conservative until calibrated on hardware.
"""
import numpy as np


class GateError(RuntimeError):
    """Raised when a safety gate rejects the plan. Abort the task; do not continue."""


class SafetyGates:
    # Angled drilling: sim may explore a bit further; hardware stays tighter.
    MAX_TILT_DEG = 25.0
    MAX_AZIM_DEG = 25.0
    REAL_MAX_TILT_DEG = 20.0
    REAL_MAX_AZIM_DEG = 20.0
    MIN_BIT_DOT_NORMAL = 0.75          # ~41 deg from board normal

    # Grasp-site workspace in world (metres), in front of the xArm base.
    WS_X = (0.18, 0.72)
    WS_Y = (-0.45, 0.35)
    WS_Z = (0.04, 0.62)

    MIN_GRIP_PTS = 20
    GRIP_RADIUS = (0.010, 0.030)
    MAX_TRACK_ERR = 0.012              # m
    MIN_JOINT_MARGIN = 0.12            # rad
    MIN_LIFT = 0.10                    # m
    MIN_LOADED_FINGERS = 3
    FINGER_LOAD = 0.4                  # N*m (sim actuator_force)
    MAX_FEED_SPEED = 0.012             # m/s along the bit during drilling
    MIN_AIM_CLEARANCE = 0.015          # m, bit tip to board face before trigger

    def __init__(self, real_robot=False):
        self.real = bool(real_robot)
        self.max_tilt = self.REAL_MAX_TILT_DEG if self.real else self.MAX_TILT_DEG
        self.max_azim = self.REAL_MAX_AZIM_DEG if self.real else self.MAX_AZIM_DEG

    def check_drill_angles(self, tilt_deg, azim_deg, bit_dir, board_n):
        if abs(tilt_deg) > self.max_tilt + 1e-6:
            raise GateError(f"tilt {tilt_deg:.1f} deg exceeds ±{self.max_tilt:.0f} deg gate")
        if abs(azim_deg) > self.max_azim + 1e-6:
            raise GateError(f"azim {azim_deg:.1f} deg exceeds ±{self.max_azim:.0f} deg gate")
        d = np.asarray(bit_dir, float); d = d / np.linalg.norm(d)
        n = np.asarray(board_n, float); n = n / np.linalg.norm(n)
        dn = float(d @ n)
        if dn < self.MIN_BIT_DOT_NORMAL:
            raise GateError(f"bit vs board normal too steep (dot={dn:.3f}, need >={self.MIN_BIT_DOT_NORMAL})")
        if d[0] <= 0:
            raise GateError("bit axis must point into the board (+X)")

    def check_workspace(self, p, name="pose"):
        p = np.asarray(p, float).reshape(3)
        lo = np.array([self.WS_X[0], self.WS_Y[0], self.WS_Z[0]])
        hi = np.array([self.WS_X[1], self.WS_Y[1], self.WS_Z[1]])
        if np.any(p < lo) or np.any(p > hi):
            raise GateError(f"{name} {np.round(p, 3)} outside workspace {lo}..{hi}")

    def check_perception(self, est, axis_hint=None):
        n_grip = int(est.n_points.get("grip", 0))
        if n_grip < self.MIN_GRIP_PTS:
            raise GateError(f"perception: only {n_grip} grip points (need {self.MIN_GRIP_PTS})")
        if not (self.GRIP_RADIUS[0] <= est.grip_radius <= self.GRIP_RADIUS[1]):
            raise GateError(f"perception: grip radius {est.grip_radius*1000:.1f} mm implausible")
        if axis_hint is not None and np.dot(est.axis[:2], axis_hint[:2]) < 0:
            raise GateError("perception: fine axis disagrees with coarse axis (sign)")

    def check_joint_margin(self, margin_rad):
        if margin_rad < self.MIN_JOINT_MARGIN:
            raise GateError(f"joint-limit margin {margin_rad:.3f} rad below {self.MIN_JOINT_MARGIN} rad")

    def check_tracking(self, err_m, name="move"):
        if err_m > self.MAX_TRACK_ERR:
            raise GateError(f"{name} tracking error {err_m*1000:.1f} mm exceeds {self.MAX_TRACK_ERR*1000:.0f} mm")

    def check_grasp(self, finger_forces, loaded_idx=(0, 1, 2, 4)):
        f = np.abs(np.asarray(finger_forces, float))
        n = int(np.sum(f[list(loaded_idx)] > self.FINGER_LOAD))
        if n < self.MIN_LOADED_FINGERS:
            raise GateError(f"grasp: {n} loaded fingers, need {self.MIN_LOADED_FINGERS}")

    def check_lift(self, lifted_m):
        if lifted_m < self.MIN_LIFT:
            raise GateError(f"lift only {lifted_m*1000:.0f} mm, drill likely slipped")

    def check_trigger_clearance(self, tip, hole_target, board_n, min_clear=None):
        min_clear = self.MIN_AIM_CLEARANCE if min_clear is None else min_clear
        n = np.asarray(board_n, float); n = n / np.linalg.norm(n)
        # signed distance from face: positive = still in front of the board
        dist = float((np.asarray(hole_target) - np.asarray(tip)) @ n)
        if dist < min_clear:
            raise GateError(f"trigger blocked: bit-to-face {dist*1000:.1f} mm < {min_clear*1000:.0f} mm")

    def feed_duration(self, distance_m, T_cmd):
        """Stretch feed time so Cartesian speed stays under MAX_FEED_SPEED."""
        vmin = abs(distance_m) / max(self.MAX_FEED_SPEED, 1e-6)
        return max(float(T_cmd), float(vmin))
