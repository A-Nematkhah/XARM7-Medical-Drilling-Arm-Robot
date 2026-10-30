"""Bounded replan decisions with hysteresis. Never swaps mid-move blindly."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from task.scene_est import SceneEstimate


@dataclass
class ReplanConfig:
    max_retries: int = 2
    drill_xy_tol_m: float = 0.025
    board_hole_tol_m: float = 0.018
    board_n_dot_min: float = 0.985
    track_err_abort: bool = True
    require_stop: bool = True


@dataclass
class ReplanDecision:
    needed: bool
    reason: str = ""
    can_retry: bool = False
    safe_to_replace: bool = False


def should_replan(
    *,
    scene_prev: SceneEstimate | None,
    scene_now: SceneEstimate,
    cfg: ReplanConfig | None = None,
    robot_moving: bool = False,
    track_err: float | None = None,
    max_track_err: float | None = None,
    retries_used: int = 0,
    path_invalid: bool = False,
) -> ReplanDecision:
    cfg = cfg or ReplanConfig()
    if robot_moving and cfg.require_stop:
        return ReplanDecision(
            False, "robot still moving; hold before replan",
            can_retry=False, safe_to_replace=False)

    if path_invalid:
        ok = retries_used < int(cfg.max_retries)
        return ReplanDecision(
            True, "planned path invalidated by scene/robot change",
            can_retry=ok, safe_to_replace=not robot_moving)

    if (cfg.track_err_abort and track_err is not None and max_track_err is not None
            and float(track_err) > float(max_track_err)):
        return ReplanDecision(
            True, f"tracking error {track_err:.4f} > {max_track_err:.4f}",
            can_retry=False, safe_to_replace=False)

    if scene_prev is None:
        return ReplanDecision(False, "no prior scene")

    if (scene_prev.drill.valid and scene_now.drill.valid
            and scene_prev.drill.position is not None
            and scene_now.drill.position is not None):
        dxy = float(np.linalg.norm(
            scene_now.drill.position[:2] - scene_prev.drill.position[:2]))
        if dxy > float(cfg.drill_xy_tol_m):
            ok = retries_used < int(cfg.max_retries)
            return ReplanDecision(
                True, f"drill XY moved {dxy*1000:.0f} mm",
                can_retry=ok, safe_to_replace=not robot_moving)

    if scene_prev.board.valid and scene_now.board.valid:
        if (scene_prev.board.hole is not None and scene_now.board.hole is not None):
            dh = float(np.linalg.norm(scene_now.board.hole - scene_prev.board.hole))
            if dh > float(cfg.board_hole_tol_m):
                ok = retries_used < int(cfg.max_retries)
                return ReplanDecision(
                    True, f"board hole moved {dh*1000:.0f} mm",
                    can_retry=ok, safe_to_replace=not robot_moving)
        if (scene_prev.board.normal is not None and scene_now.board.normal is not None):
            n0 = scene_prev.board.normal / (np.linalg.norm(scene_prev.board.normal) + 1e-12)
            n1 = scene_now.board.normal / (np.linalg.norm(scene_now.board.normal) + 1e-12)
            if float(n0 @ n1) < float(cfg.board_n_dot_min):
                ok = retries_used < int(cfg.max_retries)
                return ReplanDecision(
                    True, "board normal changed beyond hysteresis",
                    can_retry=ok, safe_to_replace=not robot_moving)

    return ReplanDecision(False, "within hysteresis")
