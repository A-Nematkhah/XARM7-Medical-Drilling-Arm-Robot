"""Held-drill relocation that never sweeps the bit through the board."""
from dataclasses import replace

import numpy as np

from robot import mat2quat
from safety import GateError

from task.constants import REORIENT_TCP_Z


def relocate_held_drill(ctx, p_goal, R_goal, profile="relocate", name="relocate",
                        final_profile=None, max_final_hop=0.0):
    """Retreat / align / reorient / approach as continuous Cartesian stages.

    Each stage is one min-jerk `move_to_pose` (sim already slerps orientation).
    Optional `max_final_hop > 0` still chops the last standoff for callers that
    need short segments; default `0` keeps the transfer continuous so the arm
    does not stop-start every ~45 mm on the way to the board.
    """
    prof = ctx.motion[profile]
    last_prof = ctx.motion[final_profile] if final_profile else prof
    try:
        slow = ctx.motion["approach"]
    except KeyError:
        slow = last_prof
    # Transit hops: keep speed, almost no settle pause between stages.
    transit = replace(prof, settle=min(float(prof.settle), 0.05),
                      t_min=min(float(prof.t_min), 0.8))
    n = np.asarray(ctx.st["board_n"], float)
    n = n / (np.linalg.norm(n) + 1e-12)
    hole = ctx.st["hole"]
    p0, R0 = ctx.robot.site_pose()
    p_goal = np.asarray(p_goal, float)
    g = ctx.gates.grasp_face_clearance(p0, hole, n)
    need = ctx.gates.REORIENT_GRASP_STANDOFF
    p_via = p0.copy()
    if g < need:
        p_via = p0 - (need - g) * n
    p_via[2] = max(float(p_via[2]), float(p0[2]), float(p_goal[2]), REORIENT_TCP_Z)
    # Slide in the face plane so the last hop is a short standoff approach, not a
    # long diagonal after the wrist has already reoriented.
    lat = p_goal - p_via
    lat = lat - n * float(lat @ n)
    p_align = p_via + lat
    p_align[2] = max(float(p_align[2]), float(p_via[2]), REORIENT_TCP_Z)
    ctx.gates.check_workspace(p_via, f"{name}-via")
    ctx.gates.check_workspace(p_align, f"{name}-align")
    ctx.gates.check_workspace(p_goal, name)
    # Keep the hand/housing off the face on every hop (finite-face samples).
    half = ctx.calib.board.half if ctx.calib is not None else None
    if ctx.tip_in_hand is not None:
        for p, R, tag in (
                (p_via, R0, f"{name}-via"),
                (p_via, R_goal, f"{name}-reorient"),
                (p_align, R0, f"{name}-align-oldR"),
                (p_align, R_goal, f"{name}-align"),
                (p_goal, R_goal, name)):
            ctx.gates.check_hand_vs_board(
                p, R, hole, n, board_half=half, name=tag)

    def _arrive_err(p, R):
        p = np.asarray(p, float)
        site_p, _ = ctx.robot.site_pose()
        return float(np.linalg.norm(site_p - p))

    def _settle_cart(p, R, vmax=0.025, settle=0.25):
        """Close leftover TCP residual. Caller gates the *final* error, not PI peak."""
        err = _arrive_err(p, R)
        for _ in range(3):
            if err <= ctx.gates.MAX_TRACK_ERR:
                break
            ctx.robot.move_to_pose(p, R, T=1.2, vmax=float(vmax), settle=float(settle))
            err = _arrive_err(p, R)
        ctx.robot.max_track_err = err
        return err

    def _gate_arrive(err, tag, label, dist, vmax, profile):
        ctx.runlog.event(name, label or "arrive", dist_mm=dist * 1000.0,
                         track_err_mm=float(err) * 1000.0, vmax=vmax, profile=profile)
        ctx.gates.check_tracking(err, tag)
        ctx.gates.check_joint_margin(ctx.robot.limit_margin(ctx.robot.get_q()))

    def _go(p, R, label, use=None, *, settle_ok=True):
        use = transit if use is None else use
        p = np.asarray(p, float)
        dist = float(np.linalg.norm(p - ctx.robot.site_pose()[0]))
        tag = f"{name}-{label}" if label else name
        ctx.gates.check_hand_vs_table(p, R, name=tag)
        kw = use.cart_kwargs(dist)
        ctx.robot.move_to_pose(p, R, **kw)
        err = _arrive_err(p, R)
        if settle_ok and err > ctx.gates.MAX_TRACK_ERR:
            err = _settle_cart(p, R, vmax=min(0.025, float(use.vmax)))
        ctx.robot.max_track_err = err
        _gate_arrive(err, tag, label, dist, use.vmax, use.name)

    def _go_split(p, R, label, use):
        """Optional chop of a long final hop; default path is one continuous move."""
        p_now = np.asarray(ctx.robot.site_pose()[0], float)
        p = np.asarray(p, float)
        dist = float(np.linalg.norm(p - p_now))
        hop = float(max_final_hop)
        if hop <= 1e-6 or dist <= hop + 1e-6:
            _go(p, R, label, use)
            return
        nseg = max(1, int(np.ceil(dist / hop)))
        for i in range(1, nseg + 1):
            pi = p_now + (i / nseg) * (p - p_now)
            tag = label if nseg == 1 else (f"{label}-{i}" if label else f"hop{i}")
            # Mid-segments: no pause; gate only the last sample of the chord.
            _go(pi, R, tag, use, settle_ok=(i == nseg))

    def _go_reorient(p, R_goal_local):
        """Rotate at fixed TCP via one continuous Cartesian slerp.

        Never joint-interpolate while holding the drill (the chord can ghost
        the bit through the board). `move_to_pose` already min-jerks R.
        """
        p = np.asarray(p, float).copy()
        p[2] = max(float(p[2]), REORIENT_TCP_Z)
        ctx.gates.check_workspace(p, f"{name}-reorient")
        p_now, R_now = ctx.robot.site_pose()
        if float(np.linalg.norm(p_now - p)) > 0.015:
            _go(p, R_now, "reorient-lift", use=transit)
            _, R_now = ctx.robot.site_pose()
        q0, q1 = mat2quat(R_now), mat2quat(R_goal_local)
        dot = float(np.clip(abs(np.dot(q0, q1)), 0.0, 1.0))
        ang = 2.0 * float(np.arccos(dot))
        # Stretch duration with angle so a large wrist flip stays smooth.
        T = max(float(transit.duration(0.0)), float(ang) / 0.7, 1.0)
        reo = replace(transit, t_min=T, settle=max(float(transit.settle), 0.08))
        ctx.gates.check_hand_vs_table(p, R_goal_local, name=f"{name}-reorient")
        ctx.robot.move_to_pose(p, R_goal_local, **reo.cart_kwargs(0.0))
        err = _arrive_err(p, R_goal_local)
        if err > ctx.gates.MAX_TRACK_ERR:
            err = _settle_cart(p, R_goal_local, vmax=min(0.025, float(slow.vmax)))
        ctx.robot.max_track_err = err
        ctx.gates.check_tracking(err, f"{name}-reorient")
        ctx.gates.check_joint_margin(ctx.robot.limit_margin(ctx.robot.get_q()))
        ctx.runlog.event(name, "reorient", dist_mm=0.0,
                         track_err_mm=err * 1000.0,
                         vmax=float(reo.vmax), profile="cart-slerp")

    if np.linalg.norm(p_via - p0) > 0.015:
        ctx.log(f"{name}: pull back {np.linalg.norm(p_via - p0)*1000:.0f} mm, then reorient")
        _go(p_via, R0, "retreat", transit)

    p_now = np.asarray(ctx.robot.site_pose()[0], float)
    _, R_now = ctx.robot.site_pose()
    # Align target from the pose we actually reached (retreat residual).
    lat = p_goal - p_now
    lat = lat - n * float(lat @ n)
    p_align = p_now + lat
    p_align[2] = max(float(p_align[2]), float(p_now[2]), REORIENT_TCP_Z)
    ctx.gates.check_workspace(p_align, f"{name}-align")
    if ctx.tip_in_hand is not None:
        ctx.gates.check_hand_vs_board(
            p_align, R_goal, hole, n, board_half=half, name=f"{name}-align")

    def _align_or_skip(p_al, R_al):
        if np.linalg.norm(np.asarray(p_al, float) - ctx.robot.site_pose()[0]) <= 0.015:
            return
        try:
            _go(p_al, R_al, "align", transit)
        except GateError as ex:
            # Optional XY slide: skip only a tracking miss. Table / board /
            # workspace / joint gates still abort — never continue into a wall.
            if "tracking" not in str(ex).lower():
                raise
            ctx.log(f"{name}-align skipped ({ex}); continue to goal")

    clear_now = ctx.gates.grasp_face_clearance(p_now, hole, n)
    # Far from the face: rotate first (AGENTS.md), then translate with R_goal.
    # Align-with-camera-R at the aim XY was the approach-far 26–28 mm tracking fail.
    if clear_now >= need:
        ctx.log(f"{name}: reorient in clear region ({clear_now*1000:.0f} mm standoff)")
        _go_reorient(p_now, R_goal)
        _align_or_skip(p_align, R_goal)
    else:
        _align_or_skip(p_align, R_now)
        _go_reorient(p_align, R_goal)
    _go_split(p_goal, R_goal, "", last_prof)
