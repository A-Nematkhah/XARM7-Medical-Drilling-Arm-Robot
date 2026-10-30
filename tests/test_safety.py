"""Unit tests for SafetyGates: angles, workspace, tracking, and angled feed."""
from types import SimpleNamespace

import numpy as np
import pytest

from safety import GateError, SafetyGates


N = np.array([1.0, 0.0, 0.0])


def _bit(tilt_deg=0.0, azim_deg=0.0, n=N, up=np.array([0.0, 0.0, 1.0])):
    """Same convention as run_drill_task.bit_direction (+tilt=up, +azim=+Y)."""
    def rodrigues(v, axis, deg):
        a = np.deg2rad(deg)
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
        return (np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)) @ v

    n = n / np.linalg.norm(n)
    d = rodrigues(n, up, azim_deg)
    tilt_axis = np.cross(d, up)
    d = rodrigues(d, tilt_axis / np.linalg.norm(tilt_axis), tilt_deg)
    return d / np.linalg.norm(d)


def test_perpendicular_angles_pass():
    g = SafetyGates()
    g.check_drill_angles(0.0, 0.0, N, N)


def test_sim_allows_moderate_tilt():
    g = SafetyGates()
    d = _bit(tilt_deg=15, azim_deg=10)
    g.check_drill_angles(15, 10, d, N, allow_angled=True)


def test_tilt_over_sim_limit_fails():
    g = SafetyGates()
    d = _bit(tilt_deg=40)
    with pytest.raises(GateError, match="tilt"):
        g.check_drill_angles(40, 0, d, N, allow_angled=True)


def test_real_robot_uses_tighter_angle_limits():
    g = SafetyGates(real_robot=True)
    d = _bit(tilt_deg=25)
    with pytest.raises(GateError, match="tilt"):
        g.check_drill_angles(25, 0, d, N, allow_angled=True)
    g.check_drill_angles(15, 10, _bit(15, 10), N, allow_angled=True)


def test_angled_locked_by_default():
    g = SafetyGates()
    d = _bit(tilt_deg=15, azim_deg=10)
    with pytest.raises(GateError, match="angled drilling locked"):
        g.check_drill_angles(15, 10, d, N)


def test_outward_bit_fails():
    g = SafetyGates()
    with pytest.raises(GateError, match="into the board"):
        g.check_drill_angles(0, 0, -N, N)


def test_steep_bit_dot_fails():
    g = SafetyGates()
    d = np.array([0.4, 0.9165, 0.0])
    d = d / np.linalg.norm(d)
    assert d @ N < g.min_dot
    with pytest.raises(GateError, match="too steep"):
        g.check_drill_angles(0, 0, d, N)


def test_workspace_box():
    g = SafetyGates()
    g.check_workspace([0.40, 0.0, 0.20], "ok")
    with pytest.raises(GateError, match="outside workspace"):
        g.check_workspace([1.5, 0.0, 0.20], "aim")


def test_hand_vs_table_is_not_tcp_box():
    """TCP z=0.05 is inside the workspace box; the LiDAR puck is not."""
    g = SafetyGates()
    R = np.eye(3)
    g.check_workspace([0.40, 0.0, 0.05], "grasp")
    with pytest.raises(GateError, match="table"):
        g.check_hand_vs_table([0.40, 0.0, 0.05], R, name="hover")
    g.check_hand_vs_table([0.40, 0.0, 0.22], R, name="hover")


def test_table_clear_loads_from_yaml():
    g = SafetyGates.from_file()
    assert g.TABLE_Z == pytest.approx(0.0)
    assert g.TABLE_CLEAR == pytest.approx(0.015)


def test_grasp_counts_load_at_threshold():
    g = SafetyGates()
    g.check_grasp([0.4, 0.4, 0.4, 0.0, 0.4, 0.0])
    with pytest.raises(GateError, match="loaded fingers"):
        g.check_grasp([0.3, 0.3, 0.3, 0.0, 0.3, 0.0])


def test_tracking_gate():
    g = SafetyGates()
    g.check_tracking(0.011, "approach")
    with pytest.raises(GateError, match="tracking"):
        g.check_tracking(0.013, "approach")


def test_transit_tip_uses_detected_hole_not_sim_origin():
    g = SafetyGates()
    hole = np.array([0.63, -0.32, 0.33])
    half = np.array([0.012, 0.16, 0.13])
    g.check_transit_tip(hole - 0.05 * N, hole, N, "ok", board_half=half)
    with pytest.raises(GateError, match="entered the board"):
        g.check_transit_tip(hole + 0.01 * N, hole, N, "reorient", board_half=half)
    # beside the face: miss the calibrated extent, so do not abort
    g.check_transit_tip(hole + 0.01 * N + np.array([0.0, 0.40, 0.0]),
                        hole, N, "beside", board_half=half)


def test_trigger_clearance():
    g = SafetyGates()
    hole = np.array([0.64, 0.0, 0.30])
    tip_ok = hole - 0.03 * N
    g.check_trigger_clearance(tip_ok, hole, N)
    with pytest.raises(GateError, match="trigger"):
        g.check_trigger_clearance(hole - 0.005 * N, hole, N)


def test_feed_along_bit_perpendicular_equals_sum():
    g = SafetyGates()
    standoff, depth, margin = 0.03, 0.025, 0.008
    feed = g.feed_along_bit(standoff, depth, margin, 1.0)
    assert feed == pytest.approx(standoff + depth + margin)


def test_feed_along_bit_divides_full_stroke_by_dn():
    """Regression: must be (standoff+depth+margin)/dn, not standoff + (depth+margin)/dn."""
    g = SafetyGates()
    standoff, depth, margin = 0.03, 0.025, 0.008
    dn = 0.5
    feed = g.feed_along_bit(standoff, depth, margin, dn)
    assert feed == pytest.approx((standoff + depth + margin) / dn)
    buggy = standoff + (depth + margin) / dn
    assert feed > buggy + 0.02
    assert feed * dn == pytest.approx(standoff + depth + margin)


def test_feed_duration_caps_speed():
    g = SafetyGates()
    # 0.063 m at 12 mm/s needs 5.25 s, so T_cmd=5 is stretched
    assert g.feed_duration(0.063, 5.0) == pytest.approx(0.063 / g.MAX_FEED_SPEED)
    assert g.feed_duration(0.01, 5.0) == pytest.approx(5.0)


def test_check_board_uses_facing_hint_not_world_x():
    g = SafetyGates()
    n = np.array([0.0, 1.0, 0.0])
    board = SimpleNamespace(n_points=500, inlier_ratio=0.8, normal=n,
                            point=np.array([0.4, 0.5, 0.3]))
    g.check_board(board, facing_hint=n)
    with pytest.raises(GateError, match="facing"):
        g.check_board(board, facing_hint=np.array([1.0, 0.0, 0.0]))


def test_check_board_without_hint_uses_point_xy():
    g = SafetyGates()
    n = np.array([0.0, 1.0, 0.0])
    board = SimpleNamespace(n_points=500, inlier_ratio=0.8, normal=n,
                            point=np.array([0.05, 0.50, 0.3]))
    g.check_board(board)


def test_perception_grip_points():
    g = SafetyGates()
    ok = SimpleNamespace(n_points={"grip": 40}, grip_radius=0.018, axis=np.array([0.0, 1.0, 0.0]))
    g.check_perception(ok, axis_hint=np.array([0.0, 1.0, 0.0]))
    bad = SimpleNamespace(n_points={"grip": 5}, grip_radius=0.018, axis=np.array([0.0, 1.0, 0.0]))
    with pytest.raises(GateError, match="grip points"):
        g.check_perception(bad)


def test_hand_vs_board_allows_beside_face_rejects_on_face():
    g = SafetyGates()
    n = np.array([1.0, 0.0, 0.0])
    hole = np.array([0.64, 0.0, 0.30])
    half = np.array([0.012, 0.16, 0.13])
    R = np.eye(3)
    # In front of the face: clear.
    g.check_hand_vs_board(hole - 0.20 * n, R, hole, n, board_half=half, name="ok")
    # Through the infinite plane but far beside the face: allowed.
    g.check_hand_vs_board(
        hole + 0.02 * n + np.array([0.0, 0.55, 0.0]), R, hole, n,
        board_half=half, name="beside")
    # Into the face slab: blocked.
    with pytest.raises(GateError, match="would hit the board"):
        g.check_hand_vs_board(hole + 0.01 * n, R, hole, n, board_half=half, name="hit")
