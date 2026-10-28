"""Phase 7: trigger, closed-loop feed to DRILL_DEPTH, retract."""
import numpy as np
from safety import GateError

from task.constants import (
    DEPTH_TOL, DRILL_DEPTH, FEED_MARGIN, HAND_GRASP, HAND_TRIGGER,
    MAX_FEED_STEP, UP,
)
from task.geom import ang_err_deg, planar_err_mm
from task.hand_io import index_angle, trigger_pressed
from task.tip import estimated_penetration, measured_tip


def drill_hole(ctx):
    ctx.phase("7/8 pull trigger + drill")
    ctx.hand.set_angles(HAND_TRIGGER, speed=500); ctx.robot.wait(1.2)
    if not trigger_pressed(ctx.hand):
        raise RuntimeError(
            f"trigger not pressed (index {index_angle(ctx.hand)}, need <= "
            f"INDEX_TRIGGER_ON)")

    dn = max(float(ctx.d @ ctx.board_n), ctx.gates.MIN_FEED_DOT)
    # hard stop: never command more than depth + small overshoot along the bit
    max_along = ctx.feed  # pre-checked by approach for housing clearance
    along = 0.0
    n_step = 0
    while True:
        tip = measured_tip(ctx)
        clear = ctx.gates.tip_clearance(tip, ctx.hole, ctx.board_n)
        pen = max(0.0, -clear)
        ctx.st["depth"] = max(float(ctx.st["depth"]), pen)
        if pen >= DRILL_DEPTH - DEPTH_TOL:
            ctx.log(f"depth reached: estimated {pen*1000:.1f} mm after {n_step} feed steps")
            break
        need = (DRILL_DEPTH - pen) / dn
        if need <= 1e-4:
            break
        step = float(min(MAX_FEED_STEP, need + DEPTH_TOL / dn, max_along - along))
        if step <= 1e-5:
            raise GateError(
                f"feed budget exhausted at estimated depth {pen*1000:.1f} mm "
                f"(target {DRILL_DEPTH*1000:.0f} mm; check FEED_MARGIN / tip)")
        along += step
        T = ctx.gates.feed_duration(step, max(0.35, step / ctx.gates.MAX_FEED_SPEED))
        ctx.robot.move_to_pose(ctx.p_aim + along * ctx.d, ctx.Rd, T=T, settle=0.05)
        # Force / dry-cut gates run in tick callbacks; re-check estimated tip here too.
        if not ctx.st["motor"]:
            ctx.gates.check_dry_cut(estimated_penetration(ctx), False, "7/8 feed")
        n_step += 1
        if n_step % 3 == 0:
            ctx.runlog.event("7/8 pull trigger + drill", "feed",
                             along_mm=along * 1000.0, depth_mm=pen * 1000.0,
                             clear_mm=clear * 1000.0)

    depth = estimated_penetration(ctx, ctx.hole, ctx.board_n)
    ctx.st["depth"] = max(float(ctx.st["depth"]), depth)
    if depth < DRILL_DEPTH - DEPTH_TOL:
        raise GateError(
            f"estimated depth only {depth*1000:.1f} mm, need {DRILL_DEPTH*1000:.0f} mm")
    if depth > DRILL_DEPTH + FEED_MARGIN + DEPTH_TOL:
        ctx.log(f"WARNING: estimated depth {depth*1000:.1f} mm overshot target "
                f"{DRILL_DEPTH*1000:.0f} mm by more than margin")

    ctx.log(f"hole depth reached {depth*1000:.1f} mm (estimated tip vs detected plane)")
    _record_accuracy(ctx)
    ctx.robot.move_to_pose(ctx.p_aim, ctx.Rd, T=2.0)
    ctx.hand.set_angles(HAND_GRASP, speed=500); ctx.robot.wait(1.0)
    ctx.robot.move_to_pose(ctx.p_aim - 0.10 * ctx.d + 0.06 * UP, ctx.Rd, T=1.5)
    ctx.phase_end("7/8 pull trigger + drill",
                  depth_mm=depth * 1000.0,
                  depth_err_mm=(depth - DRILL_DEPTH) * 1000.0,
                  entry_err_vs_gt_mm=ctx.metrics.get("entry_err_vs_gt_mm"),
                  feed_steps=n_step, feed_along_mm=along * 1000.0)


def _gt_penetration(ctx):
    """Eval-only: MuJoCo bit_tip site vs the GT board face. None on --backend real."""
    if ctx.data is None or ctx.tip_site is None or int(ctx.tip_site) < 0:
        return None, None
    tip = ctx.data.site_xpos[ctx.tip_site].copy()
    n_gt = ctx.board_n_hint
    return float((tip - ctx.hole_hint) @ n_gt), tip


def _record_accuracy(ctx):
    depth = estimated_penetration(ctx, ctx.hole, ctx.board_n)
    pen_gt, tip_gt = _gt_penetration(ctx)
    src = getattr(ctx.board, "target_source", "") or ctx.metrics.get("target_source", "")
    if pen_gt is None:
        ctx.metrics.update(
            depth_mm=round(depth * 1000.0, 2),
            depth_err_mm=round((depth - DRILL_DEPTH) * 1000.0, 2),
            target_source=src,
        )
        ctx.log(f"eval: depth est {depth*1000:.1f} mm (no sim GT on this backend)")
        ctx.runlog.event("7/8 pull trigger + drill", "eval",
                         target_source=src, depth_mm=depth * 1000.0,
                         depth_err_mm=(depth - DRILL_DEPTH) * 1000.0)
        return
    n_gt = ctx.board_n_hint
    hit_gt = tip_gt - pen_gt * n_gt
    err_gt, ey, ez = planar_err_mm(hit_gt, ctx.hole_hint, n_gt)
    err_cmd, _, _ = planar_err_mm(hit_gt, ctx.hole, ctx.board_n)
    err_mark = err_cmd
    Rdr = ctx.data.body("drill").xmat.reshape(3, 3)
    bit_actual = Rdr[:, 0]
    ctx.metrics.update(
        entry_err_vs_cmd_mm=round(err_cmd, 2),
        entry_err_vs_gt_mm=round(err_gt, 2),
        entry_err_vs_marker_mm=round(err_mark, 2),
        entry_err_y_mm=round(ey, 2),
        entry_err_z_mm=round(ez, 2),
        bit_angle_err_deg=round(ang_err_deg(bit_actual, ctx.d), 2),
        depth_mm=round(depth * 1000.0, 2),
        depth_err_mm=round((depth - DRILL_DEPTH) * 1000.0, 2),
        depth_gt_mm=round(max(0.0, pen_gt) * 1000.0, 2),
        target_source=src,
    )
    ctx.log(f"eval: entry vs GT {err_gt:.1f} mm (Y {ey:+.1f}, Z {ez:+.1f}), "
            f"vs {src or 'cmd'} {err_cmd:.1f} mm, bit angle {ctx.metrics['bit_angle_err_deg']} deg, "
            f"depth est {depth*1000:.1f} mm / GT {max(0.0, pen_gt)*1000:.1f} mm "
            f"(err {(depth-DRILL_DEPTH)*1000:+.1f})")
    ctx.runlog.event("7/8 pull trigger + drill", "eval",
                     entry_vs_gt_mm=err_gt, entry_vs_cmd_mm=err_cmd,
                     entry_vs_marker_mm=err_mark, target_source=src,
                     depth_mm=depth * 1000.0, depth_err_mm=(depth - DRILL_DEPTH) * 1000.0,
                     depth_gt_mm=max(0.0, pen_gt) * 1000.0)
