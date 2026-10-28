"""Load hand-eye / board priors and the perpendicular-drill milestone flag.

Do not treat MuJoCo BOARD_POS / CAM_POS as ground truth on hardware — put
measured values in config/calib.yaml instead.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - optional until PyYAML is installed
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIB = ROOT / "config" / "calib.yaml"
EXAMPLE_CALIB = ROOT / "config" / "calib.example.yaml"
ANGLED_UNLOCK = ROOT / "config" / "angled_unlocked"
# Match sim/scene.py BOARD_HALF (face Y/Z scaled by sqrt(1.5)).
_DEFAULT_BOARD_HALF = np.array([0.012, 0.13 * np.sqrt(1.5), 0.11 * np.sqrt(1.5)])
_UP = np.array([0.0, 0.0, 1.0])


def _as_vec3(v, default):
    if v is None:
        return np.asarray(default, float)
    a = np.asarray(v, float).reshape(-1)
    if a.size != 3:
        raise ValueError(f"expected 3-vector, got {v!r}")
    return a


def _as_vec2(v, default):
    if v is None:
        return np.asarray(default, float)
    a = np.asarray(v, float).reshape(-1)
    if a.size != 2:
        raise ValueError(f"expected 2-vector, got {v!r}")
    return a


# Camera origin in grasp_site / TCP (measured from the sim mount; replace on hardware).
_DEFAULT_HAND_EYE_T = np.array([-0.0254, 0.072, -0.0975])
_DEFAULT_HAND_EYE_RPY = np.array([0.0, -90.0, 0.0])
_DEFAULT_BIT_REACH = 0.2075
# Housing cylinder centre is 4 mm behind the grip axis (drill.xml body @ -0.004).
_DEFAULT_HOUSING_TO_TIP = 0.2115
_DEFAULT_CHUCK_BEHIND_TIP = 0.094   # tip → chuck origin along the bit
_DEFAULT_CHUCK_R = 0.018
_DEFAULT_MIN_Z = 0.105              # D435i stereo min range (m)


def intersect_ray_plane(origin, direction, center, normal):
    """Hit point of a ray with a plane. `normal` need not be unit."""
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    den = float(d @ n)
    if abs(den) < 1e-9:
        raise RuntimeError("ray parallel to board plane")
    t = float((np.asarray(center, float) - np.asarray(origin, float)) @ n) / den
    if t <= 1e-6:
        raise RuntimeError("board plane is not in front of the camera")
    return np.asarray(origin, float) + t * d


def rpy_deg_to_R(rpy_deg):
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll), angles in degrees."""
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, float).reshape(3))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


@dataclass
class HandEyeCalib:
    """Camera pose in the grasp/TCP frame (metres, RPY degrees)."""
    t: np.ndarray = field(default_factory=lambda: _DEFAULT_HAND_EYE_T.copy())
    rpy_deg: np.ndarray = field(default_factory=lambda: _DEFAULT_HAND_EYE_RPY.copy())


@dataclass
class ToolCalib:
    """Held-drill geometry that perception cannot always see (thin bit tip).

    `bit_reach` is grip-axis → tip (grasp planning). `housing_to_tip` is the
    housing-axis centre → tip used for the live tip estimate — the housing is
    a more stable length origin than the handle circle fit.
    """
    bit_reach: float = _DEFAULT_BIT_REACH
    housing_to_tip: float = _DEFAULT_HOUSING_TO_TIP
    # TCP x → tip after the RH56 has seated the handle. Measure once on a
    # fixture (sim default: 5 mm shorter than free bit_reach from palm offset).
    held_bit_reach: float = 0.202
    chuck_behind_tip: float = _DEFAULT_CHUCK_BEHIND_TIP
    chuck_r: float = _DEFAULT_CHUCK_R


@dataclass
class HandCalib:
    """Map `get_forces()` into the same units as `gates.finger_load`."""
    force_scale: float = 1.0


@dataclass
class CameraCalib:
    """Wrist RGB-D realism knobs (latency is wall-clock delay of returned frames)."""
    latency_s: float = 0.067          # ~2 frames at 30 Hz
    dropout: float = 0.02             # fraction of depth pixels zeroed
    edge_noise_m: float = 0.003       # extra σ near depth discontinuities
    min_z: float = _DEFAULT_MIN_Z     # stereo invalid below this (D435i ~105 mm)


def camera_pose_from_tcp(p_ee, R_ee, hand_eye: HandEyeCalib):
    """World camera pose from TCP FK and T_ee_cam."""
    R_he = rpy_deg_to_R(hand_eye.rpy_deg)
    R_ee = np.asarray(R_ee, float)
    p_ee = np.asarray(p_ee, float)
    R_cam = R_ee @ R_he
    p_cam = p_ee + R_ee @ np.asarray(hand_eye.t, float)
    return p_cam, R_cam


def tcp_from_camera(p_cam, R_cam, hand_eye: HandEyeCalib):
    """Inverse of `camera_pose_from_tcp` (view planning)."""
    R_he = rpy_deg_to_R(hand_eye.rpy_deg)
    R_ee = np.asarray(R_cam, float) @ R_he.T
    p_ee = np.asarray(p_cam, float) - R_ee @ np.asarray(hand_eye.t, float)
    return p_ee, R_ee


@dataclass
class BoardCalib:
    """Board prior used only as a search hint / fallback, not live aim GT."""
    position: np.ndarray = field(default_factory=lambda: np.array([0.64, -0.34, 0.34]))
    normal: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    hole_yz: np.ndarray = field(default_factory=lambda: np.array([0.02, -0.01]))
    half: np.ndarray = field(default_factory=lambda: _DEFAULT_BOARD_HALF.copy())


def board_axes(normal, up=_UP):
    """Orthonormal n (into board), y, z (in the face) from a measured/prior normal."""
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    u = np.asarray(up, float)
    z = u - n * float(u @ n)
    if np.linalg.norm(z) < 1e-6:
        z = np.array([0.0, 1.0, 0.0])
        z = z - n * float(z @ n)
    z = z / (np.linalg.norm(z) + 1e-12)
    y = np.cross(z, n)
    y = y / (np.linalg.norm(y) + 1e-12)
    return n, y, z


def hole_from_prior(board: BoardCalib):
    """Pierce-point hint from the calibrated board centre, half-thickness, and Y/Z offset."""
    n, y, z = board_axes(board.normal)
    hy, hz = (float(board.hole_yz[0]), float(board.hole_yz[1]))
    return (np.asarray(board.position, float)
            - float(board.half[0]) * n
            + hy * y
            + hz * z)


def on_board_face(hit, hole, board_n, board_half, margin=0.01):
    """True if `hit` lies on the calibrated face around the (detected) pierce point."""
    n = np.asarray(board_n, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    planar = np.asarray(hit, float) - np.asarray(hole, float)
    planar = planar - n * float(planar @ n)
    r = float(np.hypot(board_half[1], board_half[2])) + float(margin)
    return float(np.linalg.norm(planar)) <= r


@dataclass
class MilestoneCalib:
    """Angled holes stay locked until perpendicular drilling is stable."""
    allow_angled: bool = False
    unlock_file: str = "config/angled_unlocked"


@dataclass
class CalibConfig:
    hand_eye: HandEyeCalib = field(default_factory=HandEyeCalib)
    board: BoardCalib = field(default_factory=BoardCalib)
    tool: ToolCalib = field(default_factory=ToolCalib)
    hand: HandCalib = field(default_factory=HandCalib)
    camera: CameraCalib = field(default_factory=CameraCalib)
    milestone: MilestoneCalib = field(default_factory=MilestoneCalib)
    source: str = ""

    def angled_unlocked(self, cli_allow: bool = False) -> bool:
        if cli_allow or self.milestone.allow_angled:
            return True
        path = Path(self.milestone.unlock_file)
        if not path.is_absolute():
            path = ROOT / path
        return path.is_file()


def _parse_dict(raw: dict | None) -> CalibConfig:
    raw = raw or {}
    he = raw.get("hand_eye") or {}
    bd = raw.get("board") or {}
    tool = raw.get("tool") or {}
    hd = raw.get("hand") or {}
    cam = raw.get("camera") or {}
    ms = raw.get("milestone") or {}
    return CalibConfig(
        hand_eye=HandEyeCalib(
            t=_as_vec3(he.get("t"), _DEFAULT_HAND_EYE_T),
            rpy_deg=_as_vec3(he.get("rpy_deg"), _DEFAULT_HAND_EYE_RPY),
        ),
        board=BoardCalib(
            position=_as_vec3(bd.get("position"), [0.64, -0.34, 0.34]),
            normal=_as_vec3(bd.get("normal"), [1.0, 0.0, 0.0]),
            hole_yz=_as_vec2(bd.get("hole_yz"), [0.02, -0.01]),
            half=_as_vec3(bd.get("half"), _DEFAULT_BOARD_HALF),
        ),
        tool=ToolCalib(
            bit_reach=float(tool["bit_reach"]) if "bit_reach" in tool else _DEFAULT_BIT_REACH,
            housing_to_tip=float(tool["housing_to_tip"]) if "housing_to_tip" in tool
            else _DEFAULT_HOUSING_TO_TIP,
            held_bit_reach=float(tool["held_bit_reach"]) if "held_bit_reach" in tool
            else 0.202,
            chuck_behind_tip=float(tool["chuck_behind_tip"]) if "chuck_behind_tip" in tool
            else _DEFAULT_CHUCK_BEHIND_TIP,
            chuck_r=float(tool["chuck_r"]) if "chuck_r" in tool else _DEFAULT_CHUCK_R,
        ),
        hand=HandCalib(
            force_scale=float(hd["force_scale"]) if "force_scale" in hd else 1.0,
        ),
        camera=CameraCalib(
            latency_s=float(cam["latency_s"]) if "latency_s" in cam else 0.067,
            dropout=float(cam["dropout"]) if "dropout" in cam else 0.02,
            edge_noise_m=float(cam["edge_noise_m"]) if "edge_noise_m" in cam else 0.003,
            min_z=float(cam["min_z"]) if "min_z" in cam else _DEFAULT_MIN_Z,
        ),
        milestone=MilestoneCalib(
            allow_angled=bool(ms.get("allow_angled", False)),
            unlock_file=str(ms.get("unlock_file", "config/angled_unlocked")),
        ),
    )


def load_calib(path: str | os.PathLike | None = None) -> CalibConfig:
    """Load YAML calib; missing file falls back to example then built-in defaults."""
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    else:
        candidates.extend([DEFAULT_CALIB, EXAMPLE_CALIB])

    for p in candidates:
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        if yaml is None:
            raise ImportError("PyYAML is required to load calib files; pip install pyyaml")
        data = yaml.safe_load(text) or {}
        cfg = _parse_dict(data)
        cfg.source = str(p)
        n = cfg.board.normal
        cfg.board.normal = n / (np.linalg.norm(n) + 1e-12)
        return cfg

    cfg = CalibConfig()
    cfg.source = "<defaults>"
    return cfg
