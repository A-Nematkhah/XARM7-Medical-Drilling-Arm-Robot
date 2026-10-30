"""Shared arm hops that stay above the table and do not joint-interpolate blind.

Live hardware has no scratch FK (`fk_site`). In that case joint chords are
refused and the hop stays Cartesian after a vertical lift.
"""
from __future__ import annotations

import numpy as np

from safety import GateError

from task.board_obs import BODY_SAMPLES
from task.constants import REORIENT_TCP_Z, TABLE_CLEAR_Z


def table_margin(gates=None, margin=TABLE_CLEAR_Z):
    if gates is not None and hasattr(gates, "TABLE_CLEAR"):
        return float(gates.TABLE_CLEAR)
    return float(margin)


def hand_min_z(p, R, samples=BODY_SAMPLES):
    pts = np.asarray(p, float) + (np.asarray(R, float) @ np.asarray(samples, float).T).T
    return float(np.min(pts[:, 2]))


def lift_tcp(robot, z_min, T=1.0):
    """Cartesian hop straight up; keep the current wrist orientation."""
    p, R = robot.site_pose()
    if float(p[2]) >= float(z_min) - 1e-3:
        return
    p_up = np.asarray(p, float).copy()
    p_up[2] = float(z_min)
    robot.move_to_pose(p_up, R, T=max(0.6, float(T)), vmax=0.12)


def joint_path_hits_table(robot, q0, q1, n=16, margin=TABLE_CLEAR_Z, gates=None):
    """True if linear joint interpolation drives the hand through the table."""
    fk = getattr(robot, "fk_site", None)
    if fk is None:
        return True
    floor = table_margin(gates, margin)
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    for k in range(int(n) + 1):
        s = k / float(n)
        p, R = fk(q0 * (1.0 - s) + q1 * s)
        if hand_min_z(p, R) < floor:
            return True
        if 0.05 < s < 0.95 and float(p[2]) < 0.10:
            return True
    return False


def joint_path_hits_board(robot, q0, q1, *, hole, board_n, gates, board_half=None,
                          tip_in_hand=None, n=16):
    """True if a joint chord ghosts the bit or hand through the board face."""
    fk = getattr(robot, "fk_site", None)
    if fk is None:
        return True
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    tip_off = None if tip_in_hand is None else np.asarray(tip_in_hand, float)
    for k in range(int(n) + 1):
        s = k / float(n)
        p, R = fk(q0 * (1.0 - s) + q1 * s)
        try:
            gates.check_hand_vs_board(
                p, R, hole, board_n, board_half=board_half, name="joint-path")
            if tip_off is not None:
                tip = p + R @ tip_off
                gates.check_transit_tip(
                    tip, hole, board_n, "joint-path", board_half=board_half)
        except GateError:
            return True
    return False


def move_cart(robot, p, R, prof):
    """Cartesian hop using a named motion profile (vmax + t_min + settle)."""
    p = np.asarray(p, float)
    dist = float(np.linalg.norm(p - robot.site_pose()[0]))
    robot.move_to_pose(p, R, **prof.cart_kwargs(dist))


def reorient_to_pose(robot, p, R, q_goal=None, *, gates=None, T_joint=2.0,
                     T_cart=2.0, z_min=REORIENT_TCP_Z):
    """Lift, then change orientation at height. Settle on the commanded TCP."""
    p = np.asarray(p, float)
    lift_tcp(robot, z_min, T=1.2)
    q0 = robot.get_q()
    used_joint = False
    if q_goal is not None and not joint_path_hits_table(
            robot, q0, q_goal, gates=gates):
        robot.move_joints(q_goal, T=T_joint)
        used_joint = True
    if not used_joint:
        robot.move_to_pose(p, R, T=T_cart, vmax=0.12)
    err = float(np.linalg.norm(robot.site_pose()[0] - p))
    if err > 0.012:
        robot.move_to_pose(p, R, T=max(0.8, 0.5 * float(T_cart)), vmax=0.08)


def retreat_high(robot, p_start, R_start, T=1.5, z_min=REORIENT_TCP_Z):
    """Abort restore: stay high, Cartesian back toward the start pose."""
    lift_tcp(robot, z_min, T=1.0)
    p_back = np.asarray(p_start, float).copy()
    p_back[2] = max(float(p_back[2]), float(z_min))
    robot.move_to_pose(p_back, R_start, T=max(1.0, float(T)), vmax=0.12)


def safe_goto_pose(robot, p, R, *, q_goal=None, gates=None, T_joint=2.5,
                   T_cart=None, z_min=REORIENT_TCP_Z):
    """Reach `(p, R)` without a table-dipping joint chord."""
    p = np.asarray(p, float)
    T_cart = float(T_joint if T_cart is None else T_cart)
    if q_goal is None:
        sols = getattr(robot, "ik_solutions", None)
        if sols is not None:
            found = list(sols(p, R))
            q_goal = found[0] if found else None
    reorient_to_pose(
        robot, p, R, q_goal, gates=gates, T_joint=T_joint, T_cart=T_cart,
        z_min=z_min)


def safe_goto_joints(robot, q_goal, *, gates=None, T=2.5, z_min=REORIENT_TCP_Z):
    """Move to a joint vector; Cartesian fallback if the chord hits the table."""
    q_goal = np.asarray(q_goal, float)
    lift_tcp(robot, z_min, T=1.2)
    if not joint_path_hits_table(robot, robot.get_q(), q_goal, gates=gates):
        robot.move_joints(q_goal, T=T)
        return
    fk = getattr(robot, "fk_site", None)
    if fk is None:
        raise GateError(
            "joint chord vs table cannot be checked (no FK); refuse joint hop")
    p, R = fk(q_goal)
    robot.move_to_pose(p, R, T=T, vmax=0.12)
