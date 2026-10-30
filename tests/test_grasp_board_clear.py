"""Grasp / view paths prefer the board-away side; LiDAR slab rejects colliding polylines."""
from types import SimpleNamespace

import numpy as np
import pytest
from safety import GateError, SafetyGates

from task.board_obs import BODY_SAMPLES, BoardObstacle, body_world, poses_hit
from task.constants import SLIDE, TABLE_CLEAR_Z, TRANSIT_TCP_Z, UP
from task.motion_ctrl import joint_path_hits_table, reorient_to_pose
from task.phases.grasp import (
    _approach_option_list, _board_xy_prior, _clipped_side_wps, _close_with_retry,
    _grasp_frames, _grasp_side_left_only, _grasp_waypoints, _hand_min_z,
    _iter_side_geoms, _poses_hit_drill, _shoulder_folded,
    _side_signs_robot_left, _side_slide_waypoints,
)


def _ctx(board_xy, hole=None, n=None, half=None):
    hole = np.array([0.64, -0.30, 0.34]) if hole is None else np.asarray(hole, float)
    n = np.array([1.0, 0.0, 0.0]) if n is None else np.asarray(n, float)
    half = np.array([0.012, 0.16, 0.13]) if half is None else np.asarray(half, float)
    return SimpleNamespace(
        coarse_map=SimpleNamespace(board_xy=np.asarray(board_xy, float), board_n=n),
        st={"hole": hole, "board_n": n},
        hole_hint=hole,
        board_n_hint=n,
        calib=SimpleNamespace(board=SimpleNamespace(half=half)),
        gates=SafetyGates(),
        log=lambda msg: None,
    )


def test_grasp_side_order_prefers_robot_left():
    """Hand stands at pg - SLIDE*sign*bz; robot left is the larger world-Y."""
    ctx = _ctx([0.64, -0.40])
    pg = np.array([0.40, 0.0, 0.10])
    bz = np.array([0.0, 1.0, 0.0])
    # offset=-SLIDE: sign=-1 → standoff y = +SLIDE (robot left).
    order = _side_signs_robot_left(ctx, pg, bz, -SLIDE)
    assert order[0] == pytest.approx(-1.0)
    assert order[1] == pytest.approx(+1.0)
    # Opposite lateral: sign=+1 puts the hand at +Y.
    order2 = _side_signs_robot_left(ctx, pg, -bz, -SLIDE)
    assert order2[0] == pytest.approx(+1.0)
    only = _grasp_side_left_only(ctx, pg, bz, -SLIDE)
    assert only == [order[0]]
    assert len(only) == 1


def test_board_xy_prior_from_lidar_map():
    ctx = _ctx([0.55, -0.20])
    assert np.allclose(_board_xy_prior(ctx), [0.55, -0.20])


def test_grasp_via_is_pulled_away_from_board():
    n = np.array([1.0, 0.0, 0.0])
    ctx = _ctx([0.64, 0.0], hole=[0.64, 0.0, 0.30], n=n)
    pg = np.array([0.40, 0.0, 0.08])
    a = np.array([1.0, 0.0, 0.0])
    bz = np.array([0.0, 1.0, 0.0])
    wps, p_slide = _grasp_waypoints(ctx, pg, a, bz, SLIDE, 0.06, 0.10)
    via = wps[0]
    assert float((p_slide - via) @ n) > 0.05
    assert via[2] >= pg[2] + 0.18
    assert float((p_slide - via) @ n) >= 0.18  # close board → extra pull


def test_grasp_via_clips_into_workspace():
    n = np.array([1.0, 0.0, 0.0])
    ctx = _ctx([0.64, -0.40], hole=[0.64, -0.40, 0.30], n=n)
    pg = np.array([0.22, -0.50, 0.08])
    a = np.array([1.0, 0.0, 0.0])
    bz = np.array([0.0, 1.0, 0.0])
    wps, _ = _grasp_waypoints(ctx, pg, a, bz, SLIDE, 0.06, 0.10)
    via = wps[0]
    lo = np.array([0.15, -0.55, 0.04])
    hi = np.array([0.75, 0.45, 0.70])
    assert np.all(via >= lo - 1e-9) and np.all(via <= hi + 1e-9)


def test_lidar_obstacle_hits_face_allows_beside():
    board = np.array([[0.64, y, z]
                      for y in np.linspace(-0.12, 0.12, 20)
                      for z in np.linspace(0.22, 0.42, 8)])
    obs = BoardObstacle.from_points(board, n=np.array([1.0, 0.0, 0.0]), margin=0.03)
    R = np.eye(3)
    assert obs.hits(body_world([0.65, 0.0, 0.32], R, BODY_SAMPLES))
    assert not obs.hits(body_world([0.40, 0.0, 0.10], R, BODY_SAMPLES))
    # Beside the face, even if "through" the infinite plane.
    assert not obs.hits(body_world([0.70, 0.55, 0.32], R, BODY_SAMPLES))


def test_path_through_board_rejected_front_path_kept():
    board = np.array([[0.64, y, z]
                      for y in np.linspace(-0.12, 0.12, 16)
                      for z in np.linspace(0.22, 0.42, 6)])
    obs = BoardObstacle.from_points(board, n=np.array([1.0, 0.0, 0.0]))
    R = np.eye(3)
    hit = [([0.50, 0.0, 0.30], R), ([0.66, 0.0, 0.30], R)]
    clear = [([0.40, 0.0, 0.12], R), ([0.45, -0.20, 0.12], R)]
    assert poses_hit(hit, R, obs, BODY_SAMPLES)
    assert not poses_hit(clear, R, obs, BODY_SAMPLES)
    assert obs.min_clearance(body_world([0.40, 0.0, 0.32], R)) > 0.15


def test_poses_hit_drill_rejects_early_sweep():
    R = np.eye(3)
    grip = np.array([0.40, 0.0])
    # Early samples on the housing (first 65%) — reject.
    hit = [([0.40, 0.0, 0.10], R)] * 6 + [([0.40, 0.20, 0.12], R)] * 4
    # Only the last third enters the radius — OK.
    clear = [([0.40, 0.22, 0.14], R)] * 7 + [([0.40, 0.02, 0.09], R)] * 3
    assert _poses_hit_drill(hit, grip, approach_frac=0.45, radius=0.045)
    assert not _poses_hit_drill(clear, grip, approach_frac=0.45, radius=0.045)


def test_body_samples_cover_proximal_wrist():
    xs = BODY_SAMPLES[:, 0]
    assert float(np.min(xs)) <= -0.16


def test_grasp_flow_is_detect_beside_rotate_grab():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "task" / "phases" / "grasp.py").read_text(
        encoding="utf-8")
    assert "2/8 look: side view" in text
    assert "plan_path_ik" in text
    assert "move_joint_path" in text
    assert "no reachable, unoccluded side view" in text
    assert "_grasp_side_left_only" in text
    assert "top_yaw_left_down" not in text
    assert "2/8 ready: left-down" not in text


def test_grasp_frames_follow_drill_yaw():
    yaw = np.deg2rad(-20.0)
    axis = np.array([np.cos(yaw), np.sin(yaw), 0.05])
    frames = _grasp_frames(axis)
    assert len(frames) >= 4
    # First frame matches flattened +axis with world-up palm.
    a, bz, Rg, lab = frames[0]
    assert lab.startswith("bit+1")
    assert abs(float(a[2])) < 1e-9
    assert float(Rg[:, 1] @ UP) > 0.9
    # Both bit directions present.
    signs = {round(float(f[0] @ a), 0) for f in frames}
    assert 1.0 in signs and -1.0 in signs
    # Palm-Y stays mostly world-up so the pinky clears the table in transit.
    for _, _, R, _ in frames:
        assert float(R[:, 1] @ UP) > 0.75


def test_side_slide_stays_above_table_until_drop():
    pg = np.array([0.45, 0.00, 0.055])
    a = np.array([1.0, 0.0, 0.0])
    bz = np.array([0.0, 1.0, 0.0])
    Rg = np.column_stack([a, UP, bz])
    wps, p_slide = _side_slide_waypoints(pg, a, bz, SLIDE, 0.06)
    assert wps[0][2] >= TRANSIT_TCP_Z - 1e-9
    assert wps[1][2] >= TRANSIT_TCP_Z - 1e-9
    assert np.allclose(wps[1][:2], wps[2][:2])  # vertical drop, same XY
    assert wps[2][2] == pytest.approx(pg[2])
    assert wps[-1][2] == pytest.approx(pg[2])
    assert p_slide[2] == pytest.approx(pg[2])
    # Pinky/LiDAR sample (~8 cm below TCP) still clears the table at the hover.
    assert _hand_min_z(wps[0], Rg) >= TABLE_CLEAR_Z
    # Flattening the hover to grip Z would drive the puck through the table.
    low = wps[0].copy()
    low[2] = pg[2]
    assert _hand_min_z(low, Rg) < TABLE_CLEAR_Z
    SafetyGates().check_hand_vs_table(wps[0], Rg, name="hover")


def test_joint_path_hits_table_without_fk():
    class _Arm:
        pass
    assert joint_path_hits_table(_Arm(), np.zeros(7), np.ones(7))


def test_reorient_lifts_before_changing_orientation():
    hops = []

    class _Arm:
        def __init__(self):
            self.p = np.array([0.50, 0.20, 0.12])
            self.R = np.eye(3)
            self.q = np.zeros(7)

        def site_pose(self):
            return self.p.copy(), self.R.copy()

        def get_q(self):
            return self.q.copy()

        def move_to_pose(self, p, R, T=1.0, settle=0.1, vmax=None, gain=0.6):
            del T, settle, vmax, gain
            hops.append(("cart", float(p[2])))
            self.p = np.asarray(p, float).copy()
            self.R = np.asarray(R, float).copy()

        def move_joints(self, q, T=2.0):
            del T
            hops.append(("joint", float(self.p[2])))
            self.q = np.asarray(q, float).copy()

        def fk_site(self, q):
            # Fake a mid-path dip through the table when interpolating joints.
            q = np.asarray(q, float)
            s = float(np.clip(np.mean(q), 0.0, 1.0))
            z = 0.28 - 0.30 * (4.0 * s * (1.0 - s))
            return np.array([0.45, 0.10, z]), np.eye(3)

    robot = _Arm()
    p_high = np.array([0.40, 0.00, TRANSIT_TCP_Z])
    reorient_to_pose(robot, p_high, np.eye(3), np.ones(7), T_joint=1.0, T_cart=1.0)
    assert hops[0] == ("cart", pytest.approx(0.28))
    assert hops[-1][0] == "cart"
    assert hops[-1][1] == pytest.approx(TRANSIT_TCP_Z)
    assert all(kind != "joint" for kind, _ in hops)


def test_scan_for_drill_spirals_when_lidar_misses():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "task" / "phases" / "grasp.py").read_text(
        encoding="utf-8")
    assert "_scan_for_drill" in text
    assert "DRILL_SCAN" in text
    assert "_assert_drill_stable" in text
    assert "drill knocked during" in text


def test_approach_far_uses_slow_final_profile():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "task" / "phases" / "approach.py").read_text(
        encoding="utf-8")
    assert 'final_profile="approach"' in text


def test_relocate_final_is_one_continuous_approach_hop():
    from types import SimpleNamespace

    from task.held import relocate_held_drill
    from task.motion_profiles import MotionProfile

    hops = []

    class _Arm:
        max_track_err = 0.0

        def __init__(self):
            self.p = np.array([0.30, 0.10, 0.40])
            self.R = np.eye(3)

        def site_pose(self):
            return self.p.copy(), self.R.copy()

        def move_to_pose(self, p, R, T=1.0, settle=0.1, vmax=None, gain=0.6):
            del T, settle, gain
            hops.append((np.asarray(p, float).copy(), None if vmax is None else float(vmax)))
            self.p = np.asarray(p, float).copy()
            self.R = np.asarray(R, float).copy()
            self.max_track_err = 0.001

        def goto_pose_joint_space(self, p, R, T=2.5, seed=None):
            del p, R, T, seed
            raise AssertionError("held drill must not joint-interpolate")

        def get_q(self):
            return np.zeros(7)

        def limit_margin(self, q):
            del q
            return 0.5

    reloc = MotionProfile("relocate", vmax=0.08, t_min=1.2, settle=0.2)
    appr = MotionProfile("approach", vmax=0.035, t_min=0.35, settle=0.08)
    ctx = SimpleNamespace(
        motion={"relocate": reloc, "approach": appr},
        st={"board_n": np.array([1.0, 0.0, 0.0]),
            "hole": np.array([0.64, 0.10, 0.34])},
        robot=_Arm(),
        gates=SafetyGates(),
        calib=None,
        tip_in_hand=None,
        runlog=SimpleNamespace(event=lambda *a, **k: None),
        log=lambda m: None,
    )
    p_goal = np.array([0.42, 0.10, 0.34])
    relocate_held_drill(
        ctx, p_goal, np.eye(3), profile="relocate", name="approach-far",
        final_profile="approach")
    assert hops
    assert np.allclose(hops[-1][0], p_goal)
    # Default: one continuous final hop at approach vmax (no 45 mm chopping).
    slow = [(p, v) for p, v in hops if v is not None and abs(v - 0.035) < 1e-9]
    assert len(slow) == 1
    assert np.allclose(slow[0][0], p_goal)


def test_relocate_optional_split_still_chops_when_requested():
    from types import SimpleNamespace

    from task.held import relocate_held_drill
    from task.motion_profiles import MotionProfile

    hops = []

    class _Arm:
        max_track_err = 0.0

        def __init__(self):
            self.p = np.array([0.30, 0.10, 0.40])
            self.R = np.eye(3)

        def site_pose(self):
            return self.p.copy(), self.R.copy()

        def move_to_pose(self, p, R, T=1.0, settle=0.1, vmax=None, gain=0.6):
            del T, settle, gain
            hops.append((np.asarray(p, float).copy(), None if vmax is None else float(vmax)))
            self.p = np.asarray(p, float).copy()
            self.R = np.asarray(R, float).copy()
            self.max_track_err = 0.001

        def goto_pose_joint_space(self, p, R, T=2.5, seed=None):
            del p, R, T, seed
            raise AssertionError("held drill must not joint-interpolate")

        def get_q(self):
            return np.zeros(7)

        def limit_margin(self, q):
            del q
            return 0.5

    reloc = MotionProfile("relocate", vmax=0.08, t_min=1.2, settle=0.2)
    appr = MotionProfile("approach", vmax=0.035, t_min=0.35, settle=0.08)
    ctx = SimpleNamespace(
        motion={"relocate": reloc, "approach": appr},
        st={"board_n": np.array([1.0, 0.0, 0.0]),
            "hole": np.array([0.64, 0.10, 0.34])},
        robot=_Arm(),
        gates=SafetyGates(),
        calib=None,
        tip_in_hand=None,
        runlog=SimpleNamespace(event=lambda *a, **k: None),
        log=lambda m: None,
    )
    p_goal = np.array([0.42, 0.10, 0.34])
    relocate_held_drill(
        ctx, p_goal, np.eye(3), profile="relocate", name="approach-far",
        final_profile="approach", max_final_hop=0.045)
    n_slow = sum(1 for _, v in hops if v is not None and abs(v - 0.035) < 1e-9)
    assert n_slow >= 2


def test_relocate_settles_cartesian_leftover_before_tracking_gate():
    """First-hop residual must be closed with Cartesian settle, not fail the 12 mm gate."""
    from types import SimpleNamespace

    from task.held import relocate_held_drill
    from task.motion_profiles import MotionProfile

    class _Arm:
        max_track_err = 0.0

        def __init__(self):
            self.p = np.array([0.20, 0.00, 0.45])
            self.R = np.eye(3)
            self._n = 0

        def site_pose(self):
            return self.p.copy(), self.R.copy()

        def move_to_pose(self, p, R, T=1.0, settle=0.1, vmax=None, gain=0.6):
            del T, settle, vmax, gain
            p = np.asarray(p, float)
            if self._n == 0 and float(np.linalg.norm(p - self.p)) > 0.02:
                self._n += 1
                self.p = p + np.array([0.026, 0.0, 0.0])
                self.max_track_err = 0.026
            else:
                self.p = p.copy()
                self.max_track_err = 0.001
            self.R = np.asarray(R, float).copy()

        def goto_pose_joint_space(self, p, R, T=2.5, seed=None):
            del p, R, T, seed
            raise AssertionError("held drill must not joint-interpolate")

        def get_q(self):
            return np.zeros(7)

        def limit_margin(self, q):
            del q
            return 0.5

    reloc = MotionProfile("relocate", vmax=0.08, t_min=1.2, settle=0.2, joint_T=2.5)
    appr = MotionProfile("approach", vmax=0.035, t_min=0.35, settle=0.08)
    ctx = SimpleNamespace(
        motion={"relocate": reloc, "approach": appr},
        st={"board_n": np.array([1.0, 0.0, 0.0]),
            "hole": np.array([0.64, 0.00, 0.34])},
        robot=_Arm(),
        gates=SafetyGates(),
        calib=None,
        tip_in_hand=None,
        runlog=SimpleNamespace(event=lambda *a, **k: None),
        log=lambda m: None,
    )
    p_goal = np.array([0.20, 0.20, 0.45])
    relocate_held_drill(
        ctx, p_goal, np.eye(3), profile="relocate", name="approach-far",
        final_profile="approach", max_final_hop=0.20)
    assert np.linalg.norm(ctx.robot.p - p_goal) <= 0.012


def test_relocate_reorient_does_not_launder_tracking():
    """A 26 mm miss at the commanded TCP must fail the 12 mm gate."""
    from types import SimpleNamespace

    from task.held import relocate_held_drill
    from task.motion_profiles import MotionProfile

    class _Arm:
        max_track_err = 0.0

        def __init__(self):
            self.p = np.array([0.20, 0.00, 0.45])
            self.R = np.eye(3)

        def site_pose(self):
            return self.p.copy(), self.R.copy()

        def move_to_pose(self, p, R, T=1.0, settle=0.1, vmax=None, gain=0.6):
            del T, settle, vmax, gain
            p = np.asarray(p, float)
            self.p = p + np.array([0.026, 0.0, 0.0])
            self.R = np.asarray(R, float).copy()
            self.max_track_err = 0.026

        def get_q(self):
            return np.zeros(7)

        def limit_margin(self, q):
            del q
            return 0.5

    reloc = MotionProfile("relocate", vmax=0.08, t_min=1.2, settle=0.2)
    appr = MotionProfile("approach", vmax=0.035, t_min=0.35, settle=0.08)
    ctx = SimpleNamespace(
        motion={"relocate": reloc, "approach": appr},
        st={"board_n": np.array([1.0, 0.0, 0.0]),
            "hole": np.array([0.64, 0.00, 0.34])},
        robot=_Arm(),
        gates=SafetyGates(),
        calib=None,
        tip_in_hand=None,
        runlog=SimpleNamespace(event=lambda *a, **k: None),
        log=lambda m: None,
    )
    with pytest.raises(GateError, match="tracking"):
        relocate_held_drill(
            ctx, np.array([0.20, 0.20, 0.45]), np.eye(3),
            profile="relocate", name="approach-far", final_profile="approach")


def test_relocate_skips_unreachable_align_and_goes_to_goal():
    from types import SimpleNamespace

    from task.held import relocate_held_drill
    from task.motion_profiles import MotionProfile

    class _Arm:
        max_track_err = 0.0

        def __init__(self):
            self.p = np.array([0.20, 0.00, 0.45])
            self.R = np.eye(3)

        def site_pose(self):
            return self.p.copy(), self.R.copy()

        def _apply(self, p, R):
            p = np.asarray(p, float)
            # Face-plane align at z=0.45 / y=0.08 is unreachable; goal z=0.40 is OK.
            if abs(p[1] - 0.08) < 1e-6 and abs(p[2] - 0.45) < 0.01:
                self.p = p + np.array([0.0, 0.020, 0.0])
                self.max_track_err = 0.020
            else:
                self.p = p.copy()
                self.max_track_err = 0.001
            self.R = np.asarray(R, float).copy()

        def move_to_pose(self, p, R, T=1.0, settle=0.1, vmax=None, gain=0.6):
            del T, settle, vmax, gain
            self._apply(p, R)

        def goto_pose_joint_space(self, p, R, T=2.5, seed=None):
            del p, R, T, seed
            raise AssertionError("held drill must not joint-interpolate")

        def get_q(self):
            return np.zeros(7)

        def limit_margin(self, q):
            del q
            return 0.5

    reloc = MotionProfile("relocate", vmax=0.08, t_min=1.2, settle=0.2, joint_T=2.5)
    appr = MotionProfile("approach", vmax=0.035, t_min=0.35, settle=0.08)
    logs = []
    ctx = SimpleNamespace(
        motion={"relocate": reloc, "approach": appr},
        st={"board_n": np.array([1.0, 0.0, 0.0]),
            "hole": np.array([0.47, 0.00, 0.34])},
        robot=_Arm(),
        gates=SafetyGates(),
        calib=None,
        tip_in_hand=None,
        runlog=SimpleNamespace(event=lambda *a, **k: None),
        log=logs.append,
    )
    p_goal = np.array([0.20, 0.08, 0.40])
    relocate_held_drill(
        ctx, p_goal, np.eye(3), profile="relocate", name="approach-far",
        final_profile="approach", max_final_hop=0.045)
    assert np.linalg.norm(ctx.robot.p - p_goal) <= 0.012
    assert any("align skipped" in str(m) for m in logs)


def test_relocate_does_not_skip_table_gate_on_align():
    from types import SimpleNamespace

    from task.held import relocate_held_drill
    from task.motion_profiles import MotionProfile

    class _Arm:
        max_track_err = 0.0

        def __init__(self):
            self.p = np.array([0.20, 0.00, 0.45])
            self.R = np.eye(3)

        def site_pose(self):
            return self.p.copy(), self.R.copy()

        def move_to_pose(self, p, R, T=1.0, settle=0.1, vmax=None, gain=0.6):
            del T, settle, vmax, gain
            self.p = np.asarray(p, float).copy()
            self.R = np.asarray(R, float).copy()
            self.max_track_err = 0.001

        def goto_pose_joint_space(self, p, R, T=2.5, seed=None):
            del p, R, T, seed
            raise AssertionError("held drill must not joint-interpolate")

        def get_q(self):
            return np.zeros(7)

        def limit_margin(self, q):
            del q
            return 0.5

    class _Gates(SafetyGates):
        def check_hand_vs_board(self, p, R, hole, n, board_half=None, name=""):
            if "align" in str(name):
                raise GateError(f"{name} would hit the table")
            super().check_hand_vs_board(p, R, hole, n, board_half=board_half, name=name)

    reloc = MotionProfile("relocate", vmax=0.08, t_min=1.2, settle=0.2, joint_T=2.5)
    appr = MotionProfile("approach", vmax=0.035, t_min=0.35, settle=0.08)
    ctx = SimpleNamespace(
        motion={"relocate": reloc, "approach": appr},
        st={"board_n": np.array([1.0, 0.0, 0.0]),
            "hole": np.array([0.64, 0.00, 0.34])},
        robot=_Arm(),
        gates=_Gates(),
        calib=None,
        tip_in_hand=np.array([0.20, 0.0, 0.0]),
        runlog=SimpleNamespace(event=lambda *a, **k: None),
        log=lambda m: None,
    )
    with pytest.raises(GateError, match="table"):
        relocate_held_drill(
            ctx, np.array([0.20, 0.08, 0.40]), np.eye(3),
            profile="relocate", name="approach-far",
            final_profile="approach", max_final_hop=0.045)


def test_shoulder_folded_detects_seed0_posture():
    assert _shoulder_folded([-1.24, -1.63, 2.5, 0.44, 0.21, 0.45, 0.74])
    assert not _shoulder_folded([0.68, 1.1, -0.46, 1.3, -1.09, 1.3, 1.18])


def test_iter_side_geoms_clip_before_via():
    ctx = _ctx([0.64, -0.40])
    pg = np.array([0.40, 0.0, 0.12])
    a = np.array([1.0, 0.0, 0.0])
    bz = np.array([0.0, 1.0, 0.0])
    kinds = [k for *_, k in _iter_side_geoms(
        ctx, pg, a, bz, -1.0, [(SLIDE, 0.06, 0.10)])]
    assert kinds[0] == "clip"
    assert "via" in kinds and "drop" in kinds
    assert kinds.index("clip") < kinds.index("via") < kinds.index("drop")


def test_relocate_joint_retreat_when_shoulder_folded():
    from task.held import relocate_held_drill
    from task.motion_profiles import MotionProfile

    class _Arm:
        max_track_err = 0.0

        def __init__(self):
            self.p = np.array([0.40, 0.00, 0.40])
            self.R = np.eye(3)
            self.q = np.array([0.0, -1.5, 0.0, 0.9, 0.0, 1.2, 0.0])
            self.used_joint = False

        def site_pose(self):
            return self.p.copy(), self.R.copy()

        def get_q(self):
            return self.q.copy()

        def limit_margin(self, q):
            del q
            return 0.5

        def ik_solutions(self, p, R, n_random=12):
            del p, R, n_random
            return [np.array([0.2, 0.4, 0.0, 1.0, 0.0, 1.2, 0.0])]

        def fk_site(self, q):
            del q
            return np.array([0.22, 0.00, 0.40]), np.eye(3)

        def move_joints(self, q, T=2.0):
            del T
            self.used_joint = True
            self.q = np.asarray(q, float).copy()
            self.p, self.R = self.fk_site(q)
            self.max_track_err = 0.001

        def move_to_pose(self, p, R, T=1.0, settle=0.1, vmax=None, gain=0.6):
            del T, settle, vmax, gain
            self.p = np.asarray(p, float).copy()
            self.R = np.asarray(R, float).copy()
            self.max_track_err = 0.001

    reloc = MotionProfile("relocate", vmax=0.08, t_min=1.2, settle=0.2)
    appr = MotionProfile("approach", vmax=0.035, t_min=0.35, settle=0.08)
    ctx = SimpleNamespace(
        motion={"relocate": reloc, "approach": appr},
        st={"board_n": np.array([1.0, 0.0, 0.0]),
            "hole": np.array([0.50, 0.00, 0.34])},
        robot=_Arm(),
        gates=SafetyGates(),
        calib=None,
        tip_in_hand=None,
        runlog=SimpleNamespace(event=lambda *a, **k: None),
        log=lambda m: None,
    )
    relocate_held_drill(
        ctx, np.array([0.22, 0.00, 0.40]), np.eye(3),
        profile="relocate", name="board-view", final_profile="approach")
    assert ctx.robot.used_joint


def test_approach_options_include_short_slides():
    classic = _approach_option_list(prefer_short=False)
    short = _approach_option_list(prefer_short=True)
    assert classic[0][0] == pytest.approx(SLIDE)
    assert any(abs(s - 0.08) < 1e-9 for s, _, _ in classic)
    assert short[0][0] == pytest.approx(0.08)
    assert short[-1][0] == pytest.approx(SLIDE) or short[-1] in classic


def test_approach_clips_plus_y_entry_into_workspace():
    ctx = _ctx([0.64, -0.40])
    pg = np.array([0.34, 0.26, 0.12])
    a = np.array([1.0, 0.0, 0.0])
    bz_s = np.array([0.0, -1.0, 0.0])
    wps, p_slide = _clipped_side_wps(ctx, pg, a, bz_s, SLIDE, 0.06, 0.10)
    hi_y = float(ctx.gates.WS_Y[1]) - 0.005
    assert wps[0][1] <= hi_y + 1e-9
    assert p_slide[1] == pytest.approx(pg[1] + SLIDE, abs=1e-6)
    assert p_slide[1] < float(ctx.gates.WS_Y[1])


def test_close_retry_skips_air_nudges_when_one_finger():
    class _Hand:
        def __init__(self):
            self.forces = np.array([0.0, 0.0, 0.5, 0.0, 0.2, 0.1])
            self.cmd = np.array([0.0, 0.0, 88.0, 1000.0, 210.0, 0.0])

        def set_angles(self, angles, speed=None):
            del angles, speed

        def get_forces(self):
            return self.forces.copy()

        def get_angles(self):
            return self.cmd.copy()

    class _Robot:
        def wait(self, t):
            del t

        def move_to_pose(self, p, R, **kw):
            del p, R, kw
            raise AssertionError("air-nudge must not run with 1 loaded finger")

        def site_pose(self):
            return np.array([0.34, 0.26, 0.12]), np.eye(3)

    from task.motion_profiles import MotionProfile

    logs = []
    ctx = SimpleNamespace(
        robot=_Robot(),
        hand=_Hand(),
        gates=SafetyGates(),
        motion={"grasp": MotionProfile("grasp", vmax=0.05, t_min=0.4, settle=0.1)},
        log=logs.append,
    )
    close = _close_with_retry(
        ctx, np.array([0.34, 0.26, 0.12]), np.eye(3),
        lateral=np.array([0.0, 1.0, 0.0]))
    assert close.n_loaded < 2
    assert any("air nudges" in str(m) for m in logs)


def _retreat_arm(q):
    hops = []

    class _Arm:
        max_track_err = 0.0

        def __init__(self):
            self.p = np.array([0.40, 0.00, 0.27])
            self.R = np.eye(3)
            self.q = np.asarray(q, float)

        def site_pose(self):
            return self.p.copy(), self.R.copy()

        def move_to_pose(self, p, R, T=1.0, settle=0.1, vmax=None, gain=0.6):
            del T, settle, gain
            hops.append((np.asarray(p, float).copy(), None if vmax is None else float(vmax)))
            self.p = np.asarray(p, float).copy()
            self.R = np.asarray(R, float).copy()
            self.max_track_err = 0.001

        def get_q(self):
            return self.q.copy()

        def limit_margin(self, q):
            del q
            return 0.5

    return hops, _Arm()


def test_relocate_unfolded_retreat_is_one_hop():
    from task.held import relocate_held_drill
    from task.motion_profiles import MotionProfile

    hops, arm = _retreat_arm(np.zeros(7))
    reloc = MotionProfile("relocate", vmax=0.08, t_min=1.2, settle=0.2)
    appr = MotionProfile("approach", vmax=0.035, t_min=0.35, settle=0.08)
    ctx = SimpleNamespace(
        motion={"relocate": reloc, "approach": appr},
        st={"board_n": np.array([1.0, 0.0, 0.0]),
            "hole": np.array([0.50, 0.00, 0.34])},
        robot=arm,
        gates=SafetyGates(),
        calib=None,
        tip_in_hand=None,
        runlog=SimpleNamespace(event=lambda *a, **k: None),
        log=lambda m: None,
    )
    relocate_held_drill(
        ctx, np.array([0.22, 0.00, 0.34]), np.eye(3),
        profile="relocate", name="board-view", final_profile="approach")
    # Unfolded: one Cartesian pull-back, not 28 mm chops.
    assert hops
    via = hops[0][0]
    assert float(np.linalg.norm(via - np.array([0.40, 0.00, 0.27]))) > 0.10


def test_relocate_splits_long_retreat_when_folded():
    from task.held import relocate_held_drill
    from task.motion_profiles import MotionProfile

    hops, arm = _retreat_arm([0.0, -1.5, 0.0, 0.9, 0.0, 1.2, 0.0])
    reloc = MotionProfile("relocate", vmax=0.08, t_min=1.2, settle=0.2)
    appr = MotionProfile("approach", vmax=0.035, t_min=0.35, settle=0.08)
    ctx = SimpleNamespace(
        motion={"relocate": reloc, "approach": appr},
        st={"board_n": np.array([1.0, 0.0, 0.0]),
            "hole": np.array([0.50, 0.00, 0.34])},
        robot=arm,
        gates=SafetyGates(),
        calib=None,
        tip_in_hand=None,
        runlog=SimpleNamespace(event=lambda *a, **k: None),
        log=lambda m: None,
    )
    relocate_held_drill(
        ctx, np.array([0.22, 0.00, 0.34]), np.eye(3),
        profile="relocate", name="board-view", final_profile="approach")
    retreat = [p for p, _ in hops if abs(p[1]) < 1e-9]
    assert len(retreat) >= 3
    steps = [float(np.linalg.norm(retreat[i] - retreat[i - 1]))
             for i in range(1, len(retreat))]
    assert all(s <= 0.035 + 1e-6 for s in steps[:3])


def test_relocate_rotates_small_angle_when_bit_is_long():
    """4 deg at 200 mm tip is ~14 mm of arc; must not skip (return dry-cut)."""
    from task.held import relocate_held_drill
    from task.motion_profiles import MotionProfile

    ang = np.radians(4.0)
    R0 = np.array([
        [np.cos(ang), -np.sin(ang), 0.0],
        [np.sin(ang),  np.cos(ang), 0.0],
        [0.0,          0.0,         1.0],
    ])
    hops = []

    class _Arm:
        max_track_err = 0.0

        def __init__(self):
            self.p = np.array([0.30, 0.10, 0.40])
            self.R = R0.copy()

        def site_pose(self):
            return self.p.copy(), self.R.copy()

        def move_to_pose(self, p, R, T=1.0, settle=0.1, vmax=None, gain=0.6):
            del T, settle, vmax, gain
            hops.append((np.asarray(p, float).copy(), np.asarray(R, float).copy()))
            self.p = np.asarray(p, float).copy()
            self.R = np.asarray(R, float).copy()
            self.max_track_err = 0.001

        def get_q(self):
            return np.zeros(7)

        def limit_margin(self, q):
            del q
            return 0.5

    reloc = MotionProfile("relocate", vmax=0.08, t_min=1.2, settle=0.2)
    appr = MotionProfile("approach", vmax=0.035, t_min=0.35, settle=0.08)
    logs = []
    ctx = SimpleNamespace(
        motion={"relocate": reloc, "approach": appr},
        st={"board_n": np.array([1.0, 0.0, 0.0]),
            "hole": np.array([0.64, 0.10, 0.34])},
        robot=_Arm(),
        gates=SafetyGates(),
        calib=None,
        tip_in_hand=np.array([0.20, 0.0, 0.0]),
        runlog=SimpleNamespace(event=lambda *a, **k: None),
        log=logs.append,
    )
    relocate_held_drill(
        ctx, np.array([0.42, 0.10, 0.34]), np.eye(3),
        profile="relocate", name="return", final_profile="approach")
    assert not any("skip reorient" in str(m) for m in logs)
    assert any(not np.allclose(R, R0, atol=1e-6) for _, R in hops)


def test_relocate_aligns_before_long_lateral_return():
    """Hole-to-table grasp must slide in the face plane, not cut a diagonal."""
    from task.held import relocate_held_drill
    from task.motion_profiles import MotionProfile

    hops = []

    class _Arm:
        max_track_err = 0.0

        def __init__(self):
            self.p = np.array([0.30, -0.30, 0.40])
            self.R = np.eye(3)

        def site_pose(self):
            return self.p.copy(), self.R.copy()

        def move_to_pose(self, p, R, T=1.0, settle=0.1, vmax=None, gain=0.6):
            del T, settle, vmax, gain
            hops.append(np.asarray(p, float).copy())
            self.p = np.asarray(p, float).copy()
            self.R = np.asarray(R, float).copy()
            self.max_track_err = 0.001

        def get_q(self):
            return np.zeros(7)

        def limit_margin(self, q):
            del q
            return 0.5

    reloc = MotionProfile("relocate", vmax=0.08, t_min=1.2, settle=0.2)
    appr = MotionProfile("approach", vmax=0.035, t_min=0.35, settle=0.08)
    ctx = SimpleNamespace(
        motion={"relocate": reloc, "approach": appr},
        st={"board_n": np.array([1.0, 0.0, 0.0]),
            "hole": np.array([0.64, -0.30, 0.34])},
        robot=_Arm(),
        gates=SafetyGates(),
        calib=None,
        tip_in_hand=np.array([0.20, 0.0, 0.0]),
        runlog=SimpleNamespace(event=lambda *a, **k: None),
        log=lambda m: None,
    )
    p_goal = np.array([0.42, 0.00, 0.34])
    relocate_held_drill(
        ctx, p_goal, np.eye(3), profile="relocate", name="return",
        final_profile="approach")
    assert any(abs(p[1] - 0.00) < 0.02 and p[0] < 0.36 for p in hops)
    assert np.allclose(hops[-1], p_goal)


def test_locate_objects_keeps_board_pts():
    from lidar import locate_objects
    drill = np.array([[0.40 + 0.01 * np.cos(t), 0.25 + 0.01 * np.sin(t), 0.10]
                      for t in np.linspace(0, 2 * np.pi, 24)])
    board = np.array([[0.64, -0.34 + y, 0.22 + z]
                      for y in np.linspace(-0.12, 0.12, 24)
                      for z in np.linspace(0.0, 0.24, 8)])
    cmap = locate_objects(np.vstack([drill, board]), origin_xy=(0.0, 0.0))
    assert cmap.board_pts is not None
    assert len(cmap.board_pts) >= 20
    obs = BoardObstacle.from_map(cmap)
    assert obs is not None
    assert abs(obs.origin[0] - 0.64) < 0.05
