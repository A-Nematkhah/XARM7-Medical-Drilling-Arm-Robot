"""Random board + face-mark spawn; GT is eval-only, not a live aim source."""
import numpy as np
import pytest

import scene
from task.context import TaskCtx
from task.phases.drill import _gt_penetration


def test_drill_clear_of_board_rejects_face_corridor():
    pos = np.array([0.64, -0.34, 0.34])
    # In front of the face, inside the slab width — too close for a side slide.
    assert scene.drill_clear_of_board([0.50, -0.34], pos, 0.0) is False
    # Offset laterally past the face, with centre distance OK.
    far = np.array([0.40, 0.20])
    assert float(np.linalg.norm(far - pos[:2])) >= scene.MIN_DRILL_BOARD_XY
    assert scene.drill_clear_of_board(far, pos, 0.0) is True


def test_sample_board_keeps_pose_and_randomizes_mark():
    a = scene.sample_board(np.random.default_rng(7))
    b = scene.sample_board(np.random.default_rng(8))
    hy_m = float(scene.BOARD_HALF[1] - scene.MARKER_INSET)
    hz_m = float(scene.BOARD_HALF[2] - scene.MARKER_INSET)
    assert np.allclose(a[0], scene.BOARD_POS)
    assert np.allclose(b[0], scene.BOARD_POS)
    assert a[2] == 0.0 and b[2] == 0.0
    assert not np.allclose(a[1], b[1])
    for hyz in (a[1], b[1]):
        assert abs(hyz[0]) <= hy_m + 1e-12
        assert abs(hyz[1]) <= hz_m + 1e-12


def test_sample_layout_drill_and_mark_jitter_on_fixed_board():
    rng = np.random.default_rng(0)
    left_edge = scene.board_left_y()
    marks = []
    for _ in range(40):
        xy, _, pos, hyz, yaw = scene.sample_layout(rng)
        assert np.allclose(pos, scene.BOARD_POS)
        assert yaw == 0.0
        assert np.all(xy >= scene.DRILL_SPAWN_LO - 1e-12)
        assert np.all(xy <= scene.DRILL_SPAWN_HI + 1e-12)
        assert float(xy[1]) > left_edge
        marks.append(np.asarray(hyz, float).copy())
    assert not np.allclose(marks[0], marks[1])


def test_build_model_shows_spawn_box_geom():
    import mujoco
    model, _ = scene.build_model()
    gid = int(model.geom("drill_spawn_box").id)
    c = scene.drill_spawn_center()
    assert abs(float(model.geom_pos[gid][0]) - float(c[0])) < 1e-9
    assert abs(float(model.geom_pos[gid][1]) - float(c[1])) < 1e-9


def test_build_model_places_marker_at_spawn():
    import mujoco
    from robot import REST_Q

    pos = np.array([0.61, -0.41, 0.36])
    hyz = np.array([-0.05, 0.07])
    yaw = np.deg2rad(-8.0)
    model, _ = scene.build_model(board_pos=pos, hole_yz=hyz, board_yaw=yaw)
    data = mujoco.MjData(model)
    data.qpos[:7] = REST_Q
    mujoco.mj_forward(model, data)
    gid = int(model.geom("target_dot").id)
    hole, _ = scene.hole_on_face(pos, hyz, yaw)
    # Disk sits a fraction of a millimetre in front of the face plane.
    assert np.linalg.norm(data.geom_xpos[gid] - hole) < 0.003
    bid = int(model.body("board").id)
    assert np.linalg.norm(data.xpos[bid] - pos) < 1e-9


def test_default_spawn_matches_legacy_constants():
    hole, n = scene.hole_on_face()
    assert np.allclose(hole, scene.HOLE_TARGET)
    assert np.allclose(n, [1.0, 0.0, 0.0])
    assert np.allclose(scene.HOLE_TARGET[:2],
                       (scene.BOARD_POS + np.array([-scene.BOARD_HALF[0],
                                                    scene.HOLE_YZ[0], 0]))[:2])


def test_gt_penetration_uses_spawn_not_calib_hint():
    class _Data:
        def __init__(self):
            self.site_xpos = {3: np.array([0.50, -0.40, 0.35])}
            self.body = lambda name: type("B", (), {"xmat": np.eye(3).ravel()})()

    ctx = TaskCtx(
        args=None, calib=None, model=object(), data=_Data(), robot=None, hand=None,
        cam=None, views=None, det=None, board_det=None, gates=None, motion=None,
        st={}, metrics={}, runlog=None, here=".",
        board_n_hint=np.array([1.0, 0.0, 0.0]),
        hole_hint=np.array([0.628, -0.32, 0.33]),
        allow_angled=False, sim_t=lambda: 0.0, xy=np.zeros(2), yaw=0.0,
        tip_site=3,
        hole_gt=np.array([0.60, -0.40, 0.35]),
        board_n_gt=np.array([1.0, 0.0, 0.0]),
    )
    pen, tip = _gt_penetration(ctx)
    assert np.allclose(tip, [0.50, -0.40, 0.35])
    assert pen == pytest.approx(-0.10)


def test_task_eval_hole_falls_back_to_hint():
    ctx = TaskCtx(
        args=None, calib=None, model=None, data=None, robot=None, hand=None,
        cam=None, views=None, det=None, board_det=None, gates=None, motion=None,
        st={}, metrics={}, runlog=None, here=".",
        board_n_hint=np.array([1.0, 0.0, 0.0]),
        hole_hint=np.array([0.1, 0.2, 0.3]),
        allow_angled=False, sim_t=lambda: 0.0, xy=np.zeros(2), yaw=0.0,
    )
    assert np.allclose(ctx.eval_hole(), [0.1, 0.2, 0.3])
    ctx.hole_gt = np.array([0.4, 0.5, 0.6])
    assert np.allclose(ctx.eval_hole(), [0.4, 0.5, 0.6])
