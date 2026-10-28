from types import SimpleNamespace

import numpy as np
from safety import SafetyGates

from task.tip import bind_tip_in_hand, estimated_penetration, measured_tip


class _Robot:
    def site_pose(self):
        return np.array([0.4, 0.0, 0.2]), np.eye(3)


def test_measured_tip_always_uses_grasp_frame():
    ctx = SimpleNamespace(
        st={"holding": True},
        data=SimpleNamespace(site_xpos={7: np.array([0.5, 0.02, 0.3])}),
        tip_site=7,
        robot=_Robot(),
        tip_in_hand=np.array([0.2, 0.0, 0.0]),
    )
    tip = measured_tip(ctx)
    assert np.allclose(tip, [0.6, 0.0, 0.2])


def test_measured_tip_from_tcp_and_tip_in_hand():
    ctx = SimpleNamespace(
        st={"holding": False},
        data=None,
        tip_site=None,
        robot=_Robot(),
        tip_in_hand=np.array([0.2, 0.01, 0.0]),
    )
    tip = measured_tip(ctx)
    assert np.allclose(tip, [0.6, 0.01, 0.2])


def test_estimated_penetration_past_detected_face():
    ctx = SimpleNamespace(
        robot=_Robot(),
        tip_in_hand=np.array([0.2, 0.0, 0.0]),
        st={"hole": np.array([0.55, 0.0, 0.2]), "board_n": np.array([1.0, 0.0, 0.0])},
        gates=SafetyGates(),
    )
    # estimated tip at x=0.6, face at x=0.55 -> 50 mm past the face
    assert abs(estimated_penetration(ctx) - 0.05) < 1e-9


def test_bind_tip_in_hand_uses_actual_tcp():
    p = np.array([0.41, 0.002, 0.21])
    R = np.eye(3)
    tip = np.array([0.6175, 0.002, 0.21])
    t = bind_tip_in_hand(p, R, tip)
    assert np.allclose(t, [0.2075, 0.0, 0.0])
