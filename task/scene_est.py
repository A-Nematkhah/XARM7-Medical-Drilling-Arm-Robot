"""Estimated workspace scene from perception + FK (no MuJoCo GT)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from task.board_obs import BoardObstacle

# Board freshness gates (previously in config/planner.yaml scene:).
MIN_BOARD_CONFIDENCE = 0.35
BOARD_MAX_AGE_S = 8.0


@dataclass
class PoseEstimate:
    position: np.ndarray | None = None
    orientation: np.ndarray | None = None
    confidence: float = 0.0
    valid: bool = False
    source: str = "none"
    stamp_s: float = 0.0
    age_s: float = 0.0
    extras: dict = field(default_factory=dict)

    def refresh_age(self, now_s: float) -> None:
        self.age_s = max(0.0, float(now_s) - float(self.stamp_s))


@dataclass
class PlaneEstimate:
    point: np.ndarray | None = None
    normal: np.ndarray | None = None
    hole: np.ndarray | None = None
    half: np.ndarray | None = None
    inlier_ratio: float = 0.0
    n_points: int = 0
    confidence: float = 0.0
    valid: bool = False
    source: str = "none"
    stamp_s: float = 0.0
    age_s: float = 0.0

    def refresh_age(self, now_s: float) -> None:
        self.age_s = max(0.0, float(now_s) - float(self.stamp_s))


@dataclass
class SceneEstimate:
    drill: PoseEstimate = field(default_factory=PoseEstimate)
    board: PlaneEstimate = field(default_factory=PlaneEstimate)
    board_obstacle: BoardObstacle | None = None
    table_z: float = 0.0
    workspace: dict = field(default_factory=dict)
    q: np.ndarray | None = None
    ee_p: np.ndarray | None = None
    ee_R: np.ndarray | None = None
    holding: bool = False
    grasp_verified: bool = False
    stamp_s: float = 0.0
    notes: list[str] = field(default_factory=list)
    obstacles_complete: bool = False

    def to_log(self) -> dict:
        return {
            "holding": self.holding,
            "grasp_verified": self.grasp_verified,
            "obstacles_complete": self.obstacles_complete,
            "drill_valid": self.drill.valid,
            "drill_conf": round(float(self.drill.confidence), 3),
            "drill_age_s": round(float(self.drill.age_s), 2),
            "drill_source": self.drill.source,
            "board_valid": self.board.valid,
            "board_conf": round(float(self.board.confidence), 3),
            "board_age_s": round(float(self.board.age_s), 2),
            "board_source": self.board.source,
            "board_inliers": round(float(self.board.inlier_ratio), 3),
            "has_lidar_slab": self.board_obstacle is not None,
            "notes": list(self.notes),
        }


def _now(ctx) -> float:
    try:
        return float(ctx.sim_t())
    except Exception:
        return 0.0


def update_scene_from_ctx(ctx, scene: SceneEstimate | None = None) -> SceneEstimate:
    """Refresh scene fields from live perception / FK on `ctx` (no MuJoCo GT)."""
    scene = scene or getattr(ctx, "scene", None) or SceneEstimate()
    now = _now(ctx)
    scene.stamp_s = now
    gates = getattr(ctx, "gates", None)
    if gates is not None:
        scene.workspace = {
            "x": tuple(gates.WS_X), "y": tuple(gates.WS_Y), "z": tuple(gates.WS_Z),
        }
        if hasattr(gates, "TABLE_Z"):
            scene.table_z = float(gates.TABLE_Z)

    robot = getattr(ctx, "robot", None)
    if robot is not None:
        try:
            scene.q = np.asarray(robot.get_q(), float).copy()
            p, R = robot.site_pose()
            scene.ee_p = np.asarray(p, float).copy()
            scene.ee_R = np.asarray(R, float).copy()
        except Exception:
            pass

    st = getattr(ctx, "st", {}) or {}
    scene.holding = bool(st.get("holding", False))

    est = getattr(ctx, "est", None)
    if est is not None:
        grip = np.asarray(getattr(est, "grip_xy", [np.nan, np.nan]), float)
        z = float(getattr(est, "housing_underside_z", 0.1))
        axis = np.asarray(getattr(est, "axis", [1.0, 0.0, 0.0]), float)
        raw_n = getattr(est, "n_grip_pts", None)
        if raw_n is None:
            raw_n = getattr(est, "n_points", 0)
        if isinstance(raw_n, dict):
            npts = int(raw_n.get("grip", 0) or 0)
        else:
            npts = int(raw_n or 0)
        conf = float(np.clip(npts / 80.0, 0.0, 1.0)) if npts else 0.55
        scene.drill = PoseEstimate(
            position=np.array([grip[0], grip[1], z], float),
            orientation=axis,
            confidence=conf,
            valid=np.isfinite(grip).all(),
            source="wrist_d435i",
            stamp_s=now,
        )
    scene.drill.refresh_age(now)

    board = getattr(ctx, "board", None)
    if board is not None:
        inliers = float(getattr(board, "inlier_ratio", 0.0) or 0.0)
        npts = int(getattr(board, "n_points", 0) or 0)
        conf = float(np.clip(0.5 * inliers + 0.5 * min(npts / 800.0, 1.0), 0.0, 1.0))
        scene.board = PlaneEstimate(
            point=np.asarray(getattr(board, "point", board.hole), float).copy(),
            normal=np.asarray(board.normal, float).copy(),
            hole=np.asarray(board.hole, float).copy(),
            half=(np.asarray(ctx.calib.board.half, float).copy()
                  if getattr(ctx, "calib", None) is not None else None),
            inlier_ratio=inliers,
            n_points=npts,
            confidence=conf,
            valid=True,
            source=str(getattr(board, "target_source", "wrist")),
            stamp_s=now,
        )
    elif st.get("hole") is not None and st.get("board_n") is not None:
        scene.board = PlaneEstimate(
            point=np.asarray(st["hole"], float).copy(),
            normal=np.asarray(st["board_n"], float).copy(),
            hole=np.asarray(st["hole"], float).copy(),
            confidence=0.15,
            valid=False,
            source="hint_prior",
            stamp_s=now,
        )
    scene.board.refresh_age(now)

    cmap = getattr(ctx, "coarse_map", None)
    scene.board_obstacle = BoardObstacle.from_map(cmap) if cmap is not None else None
    scene.obstacles_complete = False
    if "lidar_coarse_only" not in scene.notes:
        scene.notes.append(
            "obstacle geometry incomplete: LiDAR slab + board plane only; "
            "unknown free-space treated with clearance margins")

    ctx.scene = scene
    return scene


def board_usable(scene: SceneEstimate) -> tuple[bool, str]:
    """Whether board estimate is fresh/confident enough to approach or drill."""
    b = scene.board
    if not b.valid:
        return False, "board estimate invalid (need wrist relocalization)"
    if b.confidence < MIN_BOARD_CONFIDENCE:
        return False, f"board confidence {b.confidence:.2f} < {MIN_BOARD_CONFIDENCE}"
    if b.age_s > BOARD_MAX_AGE_S:
        return False, f"board observation age {b.age_s:.1f}s exceeds {BOARD_MAX_AGE_S}s"
    return True, "ok"
