import numpy as np
import pytest

from task.target_filter import TargetFilter, _project_to_plane


def test_seed_sets_hole_and_normal():
    f = TargetFilter()
    h = f.seed([0.64, -0.32, 0.33], [2.0, 0.0, 0.0])
    assert np.allclose(h, [0.64, -0.32, 0.33])
    assert np.allclose(f.normal, [1.0, 0.0, 0.0])
    assert f.seeded()


def test_small_update_blends_toward_measurement():
    f = TargetFilter(alpha=0.5, max_jump=0.05)
    f.seed([0.0, 0.0, 0.0])
    hole, jump, ok = f.update_hole([0.010, 0.0, 0.0])
    assert ok
    assert jump == pytest.approx(0.010)
    assert hole[0] == pytest.approx(0.005)
    assert f.n_updates == 1


def test_teleport_is_rejected():
    f = TargetFilter(alpha=0.5, max_jump=0.05)
    f.seed([0.0, 0.0, 0.0])
    hole, jump, ok = f.update_hole([0.080, 0.0, 0.0])
    assert not ok
    assert hole is None
    assert jump == pytest.approx(0.080)
    assert np.allclose(f.hole, [0.0, 0.0, 0.0])
    assert f.n_rejects == 1


def test_hole_stays_on_plane_after_blend():
    n = np.array([1.0, 0.0, 0.0])
    point = np.array([0.64, 0.0, 0.0])
    f = TargetFilter(alpha=1.0, max_jump=0.05)
    f.seed([0.64, -0.02, 0.01], n)
    # measurement pulled 5 mm off the face (+X) and 4 mm in Y
    hole, _, ok = f.update_hole([0.645, 0.02, 0.01], plane_point=point, plane_n=n)
    assert ok
    assert abs((hole - point) @ n) < 1e-9
    assert hole[1] == pytest.approx(0.02)
    assert hole[2] == pytest.approx(0.01)


def test_normal_blend_and_flip_reject():
    f = TargetFilter(n_alpha=0.5, min_n_dot=0.97)
    f.seed([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    n = f.update_normal([0.999, 0.04, 0.0])
    assert n is not None
    assert n[0] > 0.99
    assert f.update_normal([-1.0, 0.0, 0.0]) is None
    assert f.n_rejects == 1


def test_project_to_plane_helper():
    q = _project_to_plane([0.70, 0.05, -0.02], [0.64, 0.0, 0.0], [1.0, 0.0, 0.0])
    assert q[0] == pytest.approx(0.64)
    assert q[1] == pytest.approx(0.05)
    assert q[2] == pytest.approx(-0.02)


def test_clip_into_board_step_blocks_face_penetration():
    from task.phases.approach import clip_into_board_step

    n = np.array([1.0, 0.0, 0.0])
    step = clip_into_board_step(np.array([0.04, 0.01, 0.0]), n, clear=0.032, min_clear=0.027)
    assert step[0] == pytest.approx(0.005)
    assert step[1] == pytest.approx(0.01)
    blocked = clip_into_board_step(np.array([0.02, 0.0, 0.0]), n, clear=0.020, min_clear=0.027)
    assert blocked[0] == pytest.approx(0.0)
