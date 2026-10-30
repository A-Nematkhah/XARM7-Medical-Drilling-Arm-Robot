"""
Safety gates between the task planner and the robot (sim today, hardware later).

Every motion that would be dangerous on a real xArm7 + RH56 + spinning drill
must pass these checks. Keep limits conservative until calibrated on hardware.

Numeric limits live in `config/gates.yaml` so hardware can retune without
editing this file. Class attributes are the fallback defaults.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from calibration import on_board_face

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATES = ROOT / "config" / "gates.yaml"


class GateError(RuntimeError):
    """Raised when a safety gate rejects the plan. Abort the task; do not continue."""


def load_gate_limits(path: str | Path | None = None, *, required: bool = False) -> dict:
    """Load the gates YAML.

    The default path may be missing (class defaults). An explicit path, or
    `required=True`, raises `GateError` instead of silently substituting.
    """
    p = Path(path) if path is not None else DEFAULT_GATES
    if yaml is None:
        if required or path is not None:
            raise GateError("PyYAML is required to load safety gates")
        return {}
    if not p.is_file():
        if required or path is not None:
            raise GateError(f"gates file not found: {p}")
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


class SafetyGates:
    # Angled drilling: sim may explore a bit further; hardware stays tighter.
    MAX_TILT_DEG = 35.0
    MAX_AZIM_DEG = 50.0
    REAL_MAX_TILT_DEG = 20.0
    REAL_MAX_AZIM_DEG = 20.0
    MIN_BIT_DOT_NORMAL = 0.50          # ~60 deg from board normal (sim)
    REAL_MIN_BIT_DOT = 0.75            # ~41 deg (hardware)

    # Grasp-site workspace in world (metres), in front of the xArm base.
    # WS_Z is TCP-only: wrap-around grasp TCP sits near 4–12 cm. Hand / LiDAR
    # hang below TCP — use check_hand_vs_table for the table, not a higher z min.
    WS_X = (0.15, 0.75)
    WS_Y = (-0.55, 0.45)
    WS_Z = (0.04, 0.70)
    TABLE_Z = 0.0
    TABLE_CLEAR = 0.015              # m, min hand/LiDAR sample height vs table

    MIN_GRIP_PTS = 20
    GRIP_RADIUS = (0.010, 0.030)
    MIN_BOARD_PTS = 300
    MIN_BOARD_INLIERS = 0.35
    MIN_BOARD_NX = 0.35              # min n · facing_hint (into-board vs robot)
    MAX_HOLE_PRIOR_ERR = 0.040         # m, detected mark vs calib/hint (catches wrong blob)
    MAX_TRACK_ERR = 0.012              # m
    MIN_JOINT_MARGIN = 0.12            # rad
    MIN_LIFT = 0.10                    # m
    MIN_LOADED_FINGERS = 3
    FINGER_LOAD = 0.4                  # N*m (sim actuator_force)
    MAX_FEED_SPEED = 0.012             # m/s along the bit during drilling
    MIN_FEED_DOT = 0.35                # clamp 1/dn so angled feed cannot explode
    MIN_AIM_CLEARANCE = 0.015          # m, bit tip to board face before trigger
    TRANSIT_MIN_CLEAR = -0.002         # m, abort only if the tip actually enters the face
    REORIENT_GRASP_STANDOFF = 0.28     # m, grasp-to-face before rotating a held drill
    BOARD_BODY_MARGIN = 0.012          # m, extra gap for hand/housing vs board plane
    HAND_FACE_MARGIN = 0.045           # m, lateral pad when hand samples skim the face
    HOUSING_R = 0.032
    HOUSING_BEHIND_TIP = 0.145       # yellow housing front, not chuck
    CHUCK_BEHIND_TIP = 0.094
    CHUCK_R = 0.018
    # Phase 5: stop on dry push / chuck contact (force-like gates).
    MAX_DRY_PEN = 0.002              # m into face without motor → abort
    MAX_BOARD_CONTACT_N = 25.0       # N; chuck↔board contact force (sim or F/T)
    MIN_CUT_SPIN = 15.0              # rad/s; stall abort while the bit is in the face
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
    # Extra wrist / pinky-LiDAR samples used only vs the table (not the board slab).
    TABLE_SAMPLES = np.array([
        [-0.02, -0.08, -0.10],  # wrist LiDAR puck (pinky side)
        [-0.12,  0.00,  0.00],
        [-0.16,  0.00,  0.00],
        [-0.18,  0.03, -0.04],
        [-0.18, -0.04, -0.04],
        [-0.14,  0.00, -0.08],
    ])

    def __init__(self, real_robot=False, limits=None):
        self.real = bool(real_robot)
        self._copy_defaults()
        if limits is not None:
            self._apply_limits(limits)
        elif DEFAULT_GATES.is_file() and yaml is not None:
            self._apply_limits(load_gate_limits(DEFAULT_GATES))
        self.max_tilt = self.REAL_MAX_TILT_DEG if self.real else self.MAX_TILT_DEG
        self.max_azim = self.REAL_MAX_AZIM_DEG if self.real else self.MAX_AZIM_DEG
        self.min_dot = self.REAL_MIN_BIT_DOT if self.real else self.MIN_BIT_DOT_NORMAL
        self.source = str(DEFAULT_GATES) if limits is None and DEFAULT_GATES.is_file() else "<defaults>"

    @classmethod
    def from_file(cls, path=None, real_robot=False):
        p = Path(path) if path is not None else DEFAULT_GATES
        limits = load_gate_limits(p, required=True)
        g = cls(real_robot=real_robot, limits=limits)
        g.source = str(p)
        return g

    def _copy_defaults(self):
        for name in (
            "MAX_TILT_DEG", "MAX_AZIM_DEG", "REAL_MAX_TILT_DEG", "REAL_MAX_AZIM_DEG",
            "MIN_BIT_DOT_NORMAL", "REAL_MIN_BIT_DOT",
            "WS_X", "WS_Y", "WS_Z", "TABLE_Z", "TABLE_CLEAR",
            "TABLE_SAMPLES",
            "MIN_GRIP_PTS", "GRIP_RADIUS", "MIN_BOARD_PTS", "MIN_BOARD_INLIERS", "MIN_BOARD_NX",
            "MAX_HOLE_PRIOR_ERR",
            "MAX_TRACK_ERR", "MIN_JOINT_MARGIN", "MIN_LIFT", "MIN_LOADED_FINGERS", "FINGER_LOAD",
            "MAX_FEED_SPEED", "MIN_FEED_DOT", "MIN_AIM_CLEARANCE", "TRANSIT_MIN_CLEAR",
            "REORIENT_GRASP_STANDOFF", "BOARD_BODY_MARGIN", "HAND_FACE_MARGIN",
            "HOUSING_R", "HOUSING_BEHIND_TIP",
            "CHUCK_BEHIND_TIP", "CHUCK_R",
            "MAX_DRY_PEN", "MAX_BOARD_CONTACT_N", "MIN_CUT_SPIN",
            "HAND_SAMPLES",
        ):
            setattr(self, name, getattr(type(self), name))

    def _apply_limits(self, raw: dict):
        sim = raw.get("sim") or {}
        real = raw.get("real") or {}
        if "max_tilt_deg" in sim:
            self.MAX_TILT_DEG = float(sim["max_tilt_deg"])
        if "max_azim_deg" in sim:
            self.MAX_AZIM_DEG = float(sim["max_azim_deg"])
        if "min_bit_dot" in sim:
            self.MIN_BIT_DOT_NORMAL = float(sim["min_bit_dot"])
        if "max_tilt_deg" in real:
            self.REAL_MAX_TILT_DEG = float(real["max_tilt_deg"])
        if "max_azim_deg" in real:
            self.REAL_MAX_AZIM_DEG = float(real["max_azim_deg"])
        if "min_bit_dot" in real:
            self.REAL_MIN_BIT_DOT = float(real["min_bit_dot"])
        ws = raw.get("workspace") or {}
        if "x" in ws:
            self.WS_X = tuple(ws["x"])
        if "y" in ws:
            self.WS_Y = tuple(ws["y"])
        if "z" in ws:
            self.WS_Z = tuple(ws["z"])
        if "table_z" in ws:
            self.TABLE_Z = float(ws["table_z"])
        if "table_clear" in ws:
            self.TABLE_CLEAR = float(ws["table_clear"])
        perc = raw.get("perception") or {}
        if "min_grip_pts" in perc:
            self.MIN_GRIP_PTS = int(perc["min_grip_pts"])
        if "grip_radius" in perc:
            self.GRIP_RADIUS = tuple(perc["grip_radius"])
        if "min_board_pts" in perc:
            self.MIN_BOARD_PTS = int(perc["min_board_pts"])
        if "min_board_inliers" in perc:
            self.MIN_BOARD_INLIERS = float(perc["min_board_inliers"])
        if "min_board_nx" in perc:
            self.MIN_BOARD_NX = float(perc["min_board_nx"])
        if "max_hole_prior_err" in perc:
            self.MAX_HOLE_PRIOR_ERR = float(perc["max_hole_prior_err"])
        mot = raw.get("motion") or {}
        mapping = {
            "max_track_err": ("MAX_TRACK_ERR", float),
            "min_joint_margin": ("MIN_JOINT_MARGIN", float),
            "min_lift": ("MIN_LIFT", float),
            "min_loaded_fingers": ("MIN_LOADED_FINGERS", int),
            "finger_load": ("FINGER_LOAD", float),
            "max_feed_speed": ("MAX_FEED_SPEED", float),
            "min_feed_dot": ("MIN_FEED_DOT", float),
            "min_aim_clearance": ("MIN_AIM_CLEARANCE", float),
            "transit_min_clear": ("TRANSIT_MIN_CLEAR", float),
            "reorient_grasp_standoff": ("REORIENT_GRASP_STANDOFF", float),
            "board_body_margin": ("BOARD_BODY_MARGIN", float),
            "hand_face_margin": ("HAND_FACE_MARGIN", float),
            "housing_r": ("HOUSING_R", float),
            "housing_behind_tip": ("HOUSING_BEHIND_TIP", float),
            "chuck_behind_tip": ("CHUCK_BEHIND_TIP", float),
            "chuck_r": ("CHUCK_R", float),
            "max_dry_pen": ("MAX_DRY_PEN", float),
            "max_board_contact_n": ("MAX_BOARD_CONTACT_N", float),
            "min_cut_spin": ("MIN_CUT_SPIN", float),
        }
        for key, (attr, cast) in mapping.items():
            if key in mot:
                setattr(self, attr, cast(mot[key]))
        if raw.get("hand_samples"):
            self.HAND_SAMPLES = np.asarray(raw["hand_samples"], float)
        if raw.get("table_samples"):
            self.TABLE_SAMPLES = np.asarray(raw["table_samples"], float)

    def check_perpendicular_milestone(self, tilt_deg, azim_deg, allow_angled=False):
        """Refuse non-zero tilt/azim until perpendicular drilling is unlocked."""
        if allow_angled:
            return
        if abs(tilt_deg) > 1e-6 or abs(azim_deg) > 1e-6:
            raise GateError(
                "angled drilling locked until perpendicular (tilt=azim=0) is stable; "
                "pass --allow-angled, run scripts/unlock_angled.py, "
                "or create config/angled_unlocked "
                "(YAML milestone.allow_angled is ignored when milestone.strict is true)")

    def check_drill_angles(self, tilt_deg, azim_deg, bit_dir, board_n, allow_angled=False):
        self.check_perpendicular_milestone(tilt_deg, azim_deg, allow_angled=allow_angled)
        if abs(tilt_deg) > self.max_tilt + 1e-6:
            raise GateError(f"tilt {tilt_deg:.1f} deg exceeds ±{self.max_tilt:.0f} deg gate")
        if abs(azim_deg) > self.max_azim + 1e-6:
            raise GateError(f"azim {azim_deg:.1f} deg exceeds ±{self.max_azim:.0f} deg gate")
        d = np.asarray(bit_dir, float); d = d / np.linalg.norm(d)
        n = np.asarray(board_n, float); n = n / np.linalg.norm(n)
        dn = float(d @ n)
        if dn <= 0:
            raise GateError("bit axis must point into the board (along the detected normal)")
        if dn < self.min_dot:
            raise GateError(f"bit vs board normal too steep (dot={dn:.3f}, need >={self.min_dot})")

    def check_workspace(self, p, name="pose"):
        p = np.asarray(p, float).reshape(3)
        lo = np.array([self.WS_X[0], self.WS_Y[0], self.WS_Z[0]])
        hi = np.array([self.WS_X[1], self.WS_Y[1], self.WS_Z[1]])
        if np.any(p < lo) or np.any(p > hi):
            raise GateError(f"{name} {np.round(p, 3)} outside workspace {lo}..{hi}")

    def check_hand_vs_table(self, p, R, name="hand"):
        """Reject poses where hand / wrist / LiDAR samples go through the table.

        `check_workspace` is TCP-only; the pinky LiDAR sits ~8 cm below TCP when
        palm Y is world-up. Transit must pass this; wrap-around grasp TCP may sit
        near WS_Z min as long as samples stay above TABLE_Z + TABLE_CLEAR.
        """
        p = np.asarray(p, float).reshape(3)
        R = np.asarray(R, float).reshape(3, 3)
        samples = np.vstack([self.HAND_SAMPLES, self.TABLE_SAMPLES])
        pts = p + (R @ samples.T).T
        zmin = float(np.min(pts[:, 2]))
        floor = float(self.TABLE_Z) + float(self.TABLE_CLEAR)
        if zmin < floor:
            raise GateError(
                f"{name} would hit the table "
                f"(sample z {zmin * 1000:.0f} mm < {floor * 1000:.0f} mm)")

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
        n = int(np.sum(f[list(loaded_idx)] >= self.FINGER_LOAD))
        if n < self.MIN_LOADED_FINGERS:
            raise GateError(f"grasp: {n} loaded fingers, need {self.MIN_LOADED_FINGERS}")

    def check_lift(self, lifted_m):
        if lifted_m < self.MIN_LIFT:
            raise GateError(f"lift only {lifted_m*1000:.0f} mm, drill likely slipped")

    def tip_clearance(self, tip, hole, board_n):
        """Positive = tip still in front of the face (metres along into-board normal)."""
        n = np.asarray(board_n, float); n = n / np.linalg.norm(n)
        return float((np.asarray(hole) - np.asarray(tip)) @ n)

    def check_board(self, board, facing_hint=None):
        if board.n_points < self.MIN_BOARD_PTS:
            raise GateError(f"board detection weak ({board.n_points} pts)")
        if board.inlier_ratio < self.MIN_BOARD_INLIERS:
            raise GateError(f"board plane inliers too few ({board.inlier_ratio:.0%})")
        n = np.asarray(board.normal, float)
        n = n / (np.linalg.norm(n) + 1e-12)
        if facing_hint is not None:
            h = np.asarray(facing_hint, float)
            h = h / (np.linalg.norm(h) + 1e-12)
        else:
            p = np.asarray(getattr(board, "point", n), float).reshape(-1)
            if p.size >= 2:
                h = np.array([float(p[0]), float(p[1]), 0.0])
                if np.linalg.norm(h) < 1e-6:
                    h = np.array([1.0, 0.0, 0.0])
                else:
                    h = h / np.linalg.norm(h)
            else:
                h = np.array([1.0, 0.0, 0.0])
        if float(n @ h) < self.MIN_BOARD_NX:
            raise GateError(
                f"board normal not facing the robot (n={np.round(n, 3)}, "
                f"dot={float(n @ h):.2f} < {self.MIN_BOARD_NX})")

    def check_hole_prior(self, hole, prior, name="hole"):
        if self.MAX_HOLE_PRIOR_ERR <= 0 or prior is None:
            return
        err = float(np.linalg.norm(np.asarray(hole, float) - np.asarray(prior, float)))
        if err > self.MAX_HOLE_PRIOR_ERR:
            raise GateError(
                f"{name} {err*1000:.1f} mm from prior (max {self.MAX_HOLE_PRIOR_ERR*1000:.0f} mm); "
                f"likely the wrong mark")

    def check_hole_on_face(self, hole, center, board_n, board_half, margin=0.01, name="hole"):
        """Mark must lie on the detected face (not a calib XYZ prior)."""
        if not on_board_face(hole, center, board_n, board_half, margin=margin):
            raise GateError(f"{name} is not on the detected board face")

    def grasp_face_clearance(self, p_grasp, hole, board_n):
        """How far the grasp site is in front of the face (m)."""
        n = np.asarray(board_n, float); n = n / np.linalg.norm(n)
        return float((np.asarray(hole) - np.asarray(p_grasp)) @ n)

    def check_transit_tip(self, tip, hole, board_n, phase="", board_half=None):
        """Chuck-board contact is excluded (so the bit can drill). During any other
        motion the tip must stay in front of the face or it will ghost through.

        `board_half` is the calibrated face size (not MuJoCo BOARD_POS). Hits
        farther from `hole` than the face diagonal are treated as a miss.
        """
        dist = self.tip_clearance(tip, hole, board_n)
        if dist >= self.TRANSIT_MIN_CLEAR:
            return
        if board_half is not None:
            n = np.asarray(board_n, float); n = n / (np.linalg.norm(n) + 1e-12)
            hit = np.asarray(tip, float) + dist * n
            if not on_board_face(hit, hole, board_n, board_half):
                return
        raise GateError(
            f"bit entered the board during {phase or 'transit'} "
            f"(clearance {dist*1000:.1f} mm)")

    def check_trigger_clearance(self, tip, hole_target, board_n, min_clear=None):
        min_clear = self.MIN_AIM_CLEARANCE if min_clear is None else min_clear
        dist = self.tip_clearance(tip, hole_target, board_n)
        if dist < min_clear:
            raise GateError(f"trigger blocked: bit-to-face {dist*1000:.1f} mm < {min_clear*1000:.0f} mm")

    def check_hand_vs_board(self, p_grasp, R, hole, board_n, board_half=None, name="hand"):
        """Reject poses where hand/wrist samples enter the board slab (finite face).

        Unlike `check_aim_clearance`, misses beside the calibrated face are allowed so
        a sideways grasp past the board edge is not blocked by the infinite plane.
        """
        n = np.asarray(board_n, float); n = n / (np.linalg.norm(n) + 1e-12)
        p = np.asarray(p_grasp, float)
        face = np.asarray(hole, float)
        pts = p + (np.asarray(R, float) @ self.HAND_SAMPLES.T).T
        face_m = float(self.HAND_FACE_MARGIN)
        for pt in pts:
            pen = float((pt - face) @ n)
            if pen <= -self.BOARD_BODY_MARGIN:
                continue
            if board_half is not None and not on_board_face(
                    pt, hole, board_n, board_half, margin=face_m):
                continue
            raise GateError(
                f"{name} would hit the board "
                f"(plane clearance {-pen*1000:.1f} mm)")

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
        chuck = tip - self.CHUCK_BEHIND_TIP * d
        chuck_pen = float((chuck - face) @ n) + self.CHUCK_R * sin_a
        if chuck_pen > -self.BOARD_BODY_MARGIN:
            raise GateError(
                f"drill chuck would hit the board "
                f"(clearance {-chuck_pen*1000:.1f} mm; try a smaller tilt/azim)")

    def check_held_clearance(self, p_grasp, Rd, tip_in_hand, hole, board_n, phase=""):
        """Geometry-only board collision (no MuJoCo ncon). Same as hardware."""
        try:
            self.check_aim_clearance(p_grasp, Rd, tip_in_hand, hole, board_n, along_d=0.0)
        except GateError as e:
            raise GateError(f"collision during {phase or 'hold'}: {e}") from e

    def check_dry_cut(self, pen_m, motor_on, phase=""):
        """Abort if the bit is pushed into the face without the chuck spinning."""
        if motor_on:
            return
        if float(pen_m) > self.MAX_DRY_PEN:
            raise GateError(
                f"dry cut aborted during {phase or 'hold'}: tip {pen_m*1000:.1f} mm into "
                f"board without motor (max {self.MAX_DRY_PEN*1000:.0f} mm)")

    def check_board_contact_force(self, force_n, phase=""):
        """Stop when chuck/housing contact force exceeds the gate (sim ncon or F/T)."""
        if float(force_n) > self.MAX_BOARD_CONTACT_N:
            raise GateError(
                f"board contact force {force_n:.1f} N exceeds "
                f"{self.MAX_BOARD_CONTACT_N:.0f} N ({phase or 'hold'})")

    def check_cut_spin(self, omega, pen_m, phase=""):
        """Abort if the chuck is stalled while the bit is in the board."""
        if float(pen_m) <= self.MAX_DRY_PEN:
            return
        if abs(float(omega)) < self.MIN_CUT_SPIN:
            rpm = abs(float(omega)) * 60.0 / (2.0 * np.pi)
            min_rpm = self.MIN_CUT_SPIN * 60.0 / (2.0 * np.pi)
            raise GateError(
                f"drill stalled during {phase or 'feed'}: chuck {rpm:.0f} rpm "
                f"with tip {float(pen_m)*1000:.1f} mm in board (need >={min_rpm:.0f} rpm)")

    def feed_along_bit(self, standoff_m, depth_m, margin_m, bit_dot_normal):
        """Travel along the bit that yields (standoff + depth + margin) along the board normal.

        Moving `s` along unit bit `d` changes face clearance by `s * (d · n)`.
        """
        dn = max(float(bit_dot_normal), self.MIN_FEED_DOT)
        return (float(standoff_m) + float(depth_m) + float(margin_m)) / dn

    def feed_duration(self, distance_m, T_cmd):
        """Stretch feed time so Cartesian speed stays under MAX_FEED_SPEED."""
        vmin = abs(distance_m) / max(self.MAX_FEED_SPEED, 1e-6)
        return max(float(T_cmd), float(vmin))
