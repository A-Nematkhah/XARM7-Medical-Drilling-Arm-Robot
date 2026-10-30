import numpy as np
import pytest

from control import (
    MIN_JERK_PEAK, CartesianPI, duration_for_distance, min_jerk,
    polyline_from_qs, q_along_polyline,
)
from hw.interfaces import ArmDriver
from hw.real_backend import RealArm, RealBackendNotReady


def test_min_jerk_ends():
    assert min_jerk(0) == pytest.approx(0.0)
    assert min_jerk(1) == pytest.approx(1.0)
    assert 0.4 < min_jerk(0.5) < 0.6


def test_joint_polyline_is_continuous():
    q0 = np.zeros(7)
    q1 = np.ones(7)
    q2 = np.ones(7) * 2
    segs = polyline_from_qs([q0, q0, q1, q2])
    assert len(segs) == 2
    total = sum(s[2] for s in segs)
    assert np.allclose(q_along_polyline(segs, 0.0), q0)
    assert np.allclose(q_along_polyline(segs, segs[0][2]), q1)
    assert np.allclose(q_along_polyline(segs, total), q2)
    assert np.allclose(q_along_polyline(segs, 0.5 * segs[0][2]), 0.5 * q1)


def test_min_jerk_peak_is_1875():
    h = 1e-6
    ds = (min_jerk(0.5 + h) - min_jerk(0.5 - h)) / (2.0 * h)
    assert ds == pytest.approx(MIN_JERK_PEAK, rel=1e-3)


def test_duration_respects_peak_vmax():
    assert duration_for_distance(0.12, 0.1, 0.04) == pytest.approx(MIN_JERK_PEAK * 3.0)
    assert duration_for_distance(0.01, 2.0, 0.04) == pytest.approx(2.0)


def test_cartesian_pi_integrates_and_freezes_when_saturated():
    pi = CartesianPI(dt=0.01, ki=1.0, i_limit=0.05)  # explicit gains for the unit test
    pi.update([0.02, 0, 0], saturated=False)
    assert pi.i_pos[0] == pytest.approx(0.0002)
    t = pi.target(np.zeros(3))
    assert t[0] == pytest.approx(0.0002)
    before = pi.i_pos.copy()
    pi.update([1.0, 0, 0], saturated=True)
    assert np.allclose(pi.i_pos, before)
    pi.reset()
    assert np.allclose(pi.i_pos, 0.0)


def test_real_arm_move_along_is_on_driver_and_needs_sdk():
    arm = RealArm(arm_ip="127.0.0.1")
    assert isinstance(arm, ArmDriver)
    with pytest.raises(RealBackendNotReady):
        arm.move_along(np.array([1.0, 0.0, 0.0]), 0.01, np.eye(3))


def test_cart_speed_never_raises_above_cap():
    from hw.real_backend import cart_speed_mm_s

    # Stall-derated 3 mm/s must stay 3, not the old 10 mm/s floor.
    assert cart_speed_mm_s(0.012, 4.0, 80.0, vmax_m_s=0.003) == pytest.approx(3.0)
    # Long hop still respects the tcp cap.
    assert cart_speed_mm_s(0.40, 1.0, 80.0) == pytest.approx(80.0)
    # SDK minimum is 1 mm/s, not 10.
    assert cart_speed_mm_s(0.0, 1.0, 80.0) == pytest.approx(1.0)


def test_safe_goto_joints_refuses_unchecked_chord():
    from safety import GateError
    from task.motion_ctrl import safe_goto_joints

    class _Arm:
        def site_pose(self):
            return np.array([0.40, 0.0, 0.30]), np.eye(3)

        def get_q(self):
            return np.zeros(7)

        def move_joints(self, q, T=1.0):
            del q, T
            raise AssertionError("must not joint-hop without FK")

        def move_to_pose(self, p, R, T=1.0, vmax=None):
            del p, R, T, vmax
            raise AssertionError("no FK pose to convert to Cartesian")

    with pytest.raises(GateError, match="no FK"):
        safe_goto_joints(_Arm(), np.ones(7))


def test_move_along_rejects_zero_direction():
    arm = RealArm(arm_ip="127.0.0.1")
    with pytest.raises(ValueError, match="zero length"):
        arm.move_along(np.zeros(3), 0.01, np.eye(3))


def test_hand_step_matches_commanded_speed():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "sim" / "robot.py").read_text(
        encoding="utf-8")
    assert "self.speed * dt * 2.0" not in text
    assert "dmax = self.speed * dt" in text


def test_task_drill_feed_uses_move_along():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "task" / "phases" / "drill.py").read_text(
        encoding="utf-8")
    assert "move_along" in text
    assert "MAX_FEED_SPEED" in text
