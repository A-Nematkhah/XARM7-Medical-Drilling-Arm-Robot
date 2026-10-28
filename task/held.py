"""Held-drill relocation that never sweeps the bit through the board."""
import numpy as np


def _duration(dist_m, t_min):
    return max(float(t_min), float(dist_m) / 0.08)


def relocate_held_drill(ctx, p_goal, R_goal, T=2.5, name="relocate"):
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
    p_via[2] = max(float(p_via[2]), float(p0[2]), float(p_goal[2]), 0.18)
    # Slide in the face plane so the last hop is a short standoff approach, not a
    # long diagonal after the wrist has already reoriented.
    lat = p_goal - p_via
    lat = lat - n * float(lat @ n)
    p_align = p_via + lat
    p_align[2] = max(float(p_align[2]), float(p_via[2]), 0.18)
    ctx.gates.check_workspace(p_via, f"{name}-via")
    ctx.gates.check_workspace(p_align, f"{name}-align")

    def _go(p, R, t_min, label):
        dist = float(np.linalg.norm(np.asarray(p, float) - ctx.robot.site_pose()[0]))
        tag = f"{name}-{label}" if label else name
        ctx.robot.move_to_pose(p, R, T=_duration(dist, t_min))
        ctx.gates.check_tracking(ctx.robot.max_track_err, tag)
        ctx.gates.check_joint_margin(ctx.robot.limit_margin(ctx.robot.get_q()))
        ctx.runlog.event(name, label or "arrive", dist_mm=dist * 1000.0,
                         track_err_mm=float(ctx.robot.max_track_err) * 1000.0)

    if np.linalg.norm(p_via - p0) > 0.015:
        ctx.log(f"{name}: pull back {np.linalg.norm(p_via - p0)*1000:.0f} mm, then reorient")
        _go(p_via, R0, 1.5, "retreat")
    if np.linalg.norm(p_align - p_via) > 0.015:
        _go(p_align, R0, 1.5, "align")
    _go(p_align, R_goal, 1.8, "reorient")
    _go(p_goal, R_goal, T, "")
