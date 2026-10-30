"""Scene estimate / board usability / replan hysteresis (no TAMP planner)."""
from types import SimpleNamespace

import numpy as np
import pytest
from safety import GateError, SafetyGates

from task.replan import should_replan
from task.scene_est import (
    PlaneEstimate, PoseEstimate, SceneEstimate, board_usable, update_scene_from_ctx,
)
from task.state_machine import TaskState, TaskStateMachine


def test_board_usable_requires_valid_confident_board():
    scene = SceneEstimate(
        board=PlaneEstimate(valid=True, confidence=0.8, age_s=0.5))
    ok, why = board_usable(scene)
    assert ok and why == "ok"
    scene.board.confidence = 0.1
    ok, why = board_usable(scene)
    assert not ok and "confidence" in why


def test_state_machine_abort_and_retry():
    sm = TaskStateMachine()
    sm.enter(TaskState.APPROACH_BOARD, reason="go")
    assert sm.state == TaskState.APPROACH_BOARD
    sm.abort("boom")
    assert sm.state == TaskState.ABORT_OR_SAFE_RECOVERY
    assert sm.fail_reason == "boom"
    sm2 = TaskStateMachine()
    assert sm2.bump_retry("approach", 2)
    assert sm2.bump_retry("approach", 2)
    assert not sm2.bump_retry("approach", 2)


def test_should_replan_on_hole_jump():
    prev = SceneEstimate(
        board=PlaneEstimate(
            valid=True, hole=np.array([0.64, -0.30, 0.34]),
            normal=np.array([1.0, 0.0, 0.0]), confidence=0.9),
        drill=PoseEstimate(valid=True, position=np.array([0.4, 0.0, 0.1])),
    )
    now = SceneEstimate(
        board=PlaneEstimate(
            valid=True, hole=np.array([0.64, -0.30, 0.34]) + np.array([0, 0.05, 0]),
            normal=np.array([1.0, 0.0, 0.0]), confidence=0.9),
        drill=PoseEstimate(valid=True, position=np.array([0.4, 0.0, 0.1])),
    )
    d = should_replan(scene_prev=prev, scene_now=now, robot_moving=False, retries_used=0)
    assert d.needed and d.can_retry


def test_should_replan_holds_while_moving():
    d = should_replan(
        scene_prev=SceneEstimate(), scene_now=SceneEstimate(),
        robot_moving=True, retries_used=0)
    assert not d.needed
    assert "moving" in d.reason


def test_tracking_error_prevents_continue():
    d = should_replan(
        scene_prev=SceneEstimate(), scene_now=SceneEstimate(),
        track_err=0.05, max_track_err=0.012, robot_moving=False)
    assert d.needed and not d.can_retry


def test_update_scene_from_board():
    board = SimpleNamespace(
        point=np.array([0.64, -0.34, 0.34]),
        normal=np.array([1.0, 0.0, 0.0]),
        hole=np.array([0.64, -0.32, 0.33]),
        inlier_ratio=0.7,
        n_points=500,
        target_source="marker",
    )
    ctx = SimpleNamespace(
        scene=None, sim_t=lambda: 1.0, gates=SafetyGates(),
        robot=None, st={"holding": False}, est=None, board=board,
        calib=SimpleNamespace(board=SimpleNamespace(half=np.array([0.012, 0.16, 0.13]))),
        coarse_map=None,
    )
    scene = update_scene_from_ctx(ctx)
    assert scene.board.valid
    ok, _ = board_usable(scene)
    assert ok


def test_angled_still_locked_by_gates():
    gates = SafetyGates()
    n = np.array([1.0, 0.0, 0.0])
    d = np.array([np.cos(np.deg2rad(15)), 0.0, np.sin(np.deg2rad(15))])
    with pytest.raises(GateError):
        gates.check_drill_angles(15.0, 0.0, d, n, allow_angled=False)


def test_real_backend_text_no_servo_step_in_live_loop():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "hw" / "real_backend.py").read_text(
        encoding="utf-8")
    assert "mujoco.mj_jacSite" not in text
    assert "robot.servo_step" not in text
