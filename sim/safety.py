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
    MAX_TILT_DEG = 35.0
    MAX_AZIM_DEG = 50.0
    REAL_MAX_TILT_DEG = 20.0
    REAL_MAX_AZIM_DEG = 20.0
    MIN_BIT_DOT_NORMAL = 0.50          # ~60 deg from board normal (sim)
    REAL_MIN_BIT_DOT = 0.75            # ~41 deg (hardware)

    # Grasp-site workspace in world (metres), in front of the xArm base.
    WS_X = (0.15, 0.75)
    WS_Y = (-0.55, 0.45)
    WS_Z = (0.04, 0.70)

    MIN_GRIP_PTS = 20
    GRIP_RADIUS = (0.010, 0.030)
    MAX_TRACK_ERR = 0.012              # m
    MIN_JOINT_MARGIN = 0.12            # rad
    MIN_LIFT = 0.10                    # m
    MIN_LOADED_FINGERS = 3
    FINGER_LOAD = 0.4                  # N*m (sim actuator_force)
    MAX_FEED_SPEED = 0.012             # m/s along the bit during drilling
    MIN_AIM_CLEARANCE = 0.015          # m, bit tip to board face before trigger
    TRANSIT_MIN_CLEAR = -0.002         # m, abort only if the tip actually enters the face
    REORIENT_GRASP_STANDOFF = 0.28     # m, grasp-to-face before rotating a held drill
    BOARD_BODY_MARGIN = 0.012          # m, extra gap for hand/housing vs board plane
    HOUSING_R = 0.032
    HOUSING_BEHIND_TIP = 0.145
    # grasp-frame samples of the hand/wrist (x=palm/bit, y=index, z=fingers)
    HAND_SAMPLES = np.array([
        [0.00, 0.00, 0.00],
        [0.00, 0.00, 0.10],
        [0.00, 0.00, 0.14],
        [0.00, 0.05, 0.10],
        [0.00, -0.04, 0.08],
        [-0.06, 0.00, 0.05],
        [-0.10, 0.03, 0.03],
    ])

    def __init__(self, real_robot=False):
        self.real = bool(real_robot)
        self.max_tilt = self.REAL_MAX_TILT_DEG if self.real else self.MAX_TILT_DEG
        self.max_azim = self.REAL_MAX_AZIM_DEG if self.real else self.MAX_AZIM_DEG
        self.min_dot = self.REAL_MIN_BIT_DOT if self.real else self.MIN_BIT_DOT_NORMAL

    def check_drill_angles(self, tilt_deg, azim_deg, bit_dir, board_n):
        if abs(tilt_deg) > self.max_tilt + 1e-6:
            raise GateError(f"tilt {tilt_deg:.1f} deg exceeds ±{self.max_tilt:.0f} deg gate")
        if abs(azim_deg) > self.max_azim + 1e-6:
            raise GateError(f"azim {azim_deg:.1f} deg exceeds ±{self.max_azim:.0f} deg gate")
        d = np.asarray(bit_dir, float); d = d / np.linalg.norm(d)
        n = np.asarray(board_n, float); n = n / np.linalg.norm(n)
        dn = float(d @ n)
        if dn < self.min_dot:
            raise GateError(f"bit vs board normal too steep (dot={dn:.3f}, need >={self.min_dot})")
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

    def tip_clearance(self, tip, hole, board_n):
        """Positive = tip still in front of the face (metres along into-board normal)."""
        n = np.asarray(board_n, float); n = n / np.linalg.norm(n)
        return float((np.asarray(hole) - np.asarray(tip)) @ n)

    def check_board(self, board):
        if board.n_points < 300:
            raise GateError(f"board detection weak ({board.n_points} pts)")
        if board.inlier_ratio < 0.35:
            raise GateError(f"board plane inliers too few ({board.inlier_ratio:.0%})")
        n = np.asarray(board.normal, float)
        if abs(n[0]) < 0.35:
            raise GateError(f"board normal not facing the robot (n={np.round(n, 3)})")

    def grasp_face_clearance(self, p_grasp, hole, board_n):
        """How far the grasp site is in front of the face (m)."""
        n = np.asarray(board_n, float); n = n / np.linalg.norm(n)
        return float((np.asarray(hole) - np.asarray(p_grasp)) @ n)

    def check_transit_tip(self, tip, hole, board_n, phase="", board_pos=None, board_half=None):
        """Chuck-board contact is excluded (so the bit can drill). During any other
        motion the tip must stay in front of the face or it will ghost through."""
        dist = self.tip_clearance(tip, hole, board_n)
        if dist >= self.TRANSIT_MIN_CLEAR:
            return
        n = np.asarray(board_n, float); n = n / np.linalg.norm(n)
        hit = np.asarray(tip, float) + dist * n
        if board_pos is not None and board_half is not None:
            rel = hit - np.asarray(board_pos, float)
            hy, hz = float(board_half[1]), float(board_half[2])
            if abs(rel[1]) > hy + 0.01 or abs(rel[2]) > hz + 0.01:
                return
        raise GateError(
            f"bit entered the board during {phase or 'transit'} "
            f"(clearance {dist*1000:.1f} mm)")

    def check_trigger_clearance(self, tip, hole_target, board_n, min_clear=None):
        min_clear = self.MIN_AIM_CLEARANCE if min_clear is None else min_clear
        dist = self.tip_clearance(tip, hole_target, board_n)
        if dist < min_clear:
            raise GateError(f"trigger blocked: bit-to-face {dist*1000:.1f} mm < {min_clear*1000:.0f} mm")

    def check_aim_clearance(self, p_grasp, Rd, tip_in_hand, hole, board_n, along_d=0.0):
        """Reject poses where the hand or drill housing would penetrate the board plane."""
        n = np.asarray(board_n, float); n = n / np.linalg.norm(n)
        d = np.asarray(Rd[:, 0], float)
        p = np.asarray(p_grasp, float) + along_d * d
        face = np.asarray(hole, float)
        pts = p + (np.asarray(Rd) @ self.HAND_SAMPLES.T).T
        hand_pen = float(np.max((pts - face) @ n))
        if hand_pen > -self.BOARD_BODY_MARGIN:
            raise GateError(
                f"hand would hit the board at this angle "
                f"(plane clearance {-hand_pen*1000:.1f} mm; try a smaller tilt/azim)")
        tip = p + np.asarray(Rd) @ np.asarray(tip_in_hand, float)
        sin_a = float(np.linalg.norm(n - (n @ d) * d))
        house = tip - self.HOUSING_BEHIND_TIP * d
        house_pen = float((house - face) @ n) + self.HOUSING_R * sin_a
        if house_pen > -self.BOARD_BODY_MARGIN:
            raise GateError(
                f"drill housing would hit the board "
                f"(clearance {-house_pen*1000:.1f} mm; try a smaller tilt/azim)")

    def feed_duration(self, distance_m, T_cmd):
        """Stretch feed time so Cartesian speed stays under MAX_FEED_SPEED."""
        vmin = abs(distance_m) / max(self.MAX_FEED_SPEED, 1e-6)
        return max(float(T_cmd), float(vmin))
