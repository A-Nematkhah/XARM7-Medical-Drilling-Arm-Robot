"""Phase 7: trigger, closed-loop feed to DRILL_DEPTH, retract."""
import numpy as np
from safety import GateError

from task.constants import (
    DEPTH_TOL, DRILL_DEPTH, FEED_MARGIN, HAND_GRASP, UP,
)
from task.feed_ctrl import DepthFeedCtrl
from task.geom import ang_err_deg, planar_err_mm
from task.grasp_ctrl import overlay_index
from task.hand_io import index_angle, trigger_pressed
from task.phases.approach import nudge_to_aim
from task.tip import estimated_penetration, log_tip_vs_gt, measured_tip, rebind_held_tip

_MAX_FEED_ITERS = 80


def drill_hole(ctx):
    ctx.phase("7/8 pull trigger + drill")
    wrap = ctx.grasp_cmd if ctx.grasp_cmd is not None else HAND_GRASP
    ctx.hand.set_angles(overlay_index(wrap, 0), speed=500)
    t0 = ctx.sim_t()
    while not ctx.st["motor"]:
        if ctx.sim_t() - t0 > 4.0:
            raise RuntimeError(
                f"motor did not latch (index {index_angle(ctx.hand)}, "
                f"pressed={trigger_pressed(ctx.hand)})")
        # Keep commanding index closed; wrap fingers stay at the grasp freeze.
        ctx.hand.set_angles(overlay_index(wrap, 0), speed=500)
        ctx.robot.wait(0.15)
    if not trigger_pressed(ctx.hand):
        raise RuntimeError(
            f"trigger not pressed (index {index_angle(ctx.hand)}, need <= "
            f"INDEX_TRIGGER_ON)")
    # Trigger squeeze seats the drill in the palm; refresh tip, then re-aim.
    rebind_held_tip(ctx, "post-trigger")
    nudge_to_aim(ctx, "post-trigger aim")
    p_aim, _ = ctx.robot.site_pose()
    ctx.p_aim = p_aim
    log_tip_vs_gt(ctx, "pre-feed")
    _log_gt_planar_at_aim(ctx)

    dn = max(float(ctx.d @ ctx.board_n), ctx.gates.MIN_FEED_DOT)
    # hard stop: never command more than depth + small overshoot along the bit
    max_along = ctx.feed  # pre-checked by approach for housing clearance
    along = 0.0
    n_step = 0
    feed = ctx.motion["feed"]
    vmax = min(feed.vmax, ctx.gates.MAX_FEED_SPEED)
    ctrl = DepthFeedCtrl.from_profile(
        feed, target=DRILL_DEPTH, vmax=vmax, dn=dn, tol=0.4 * DEPTH_TOL)
    dt = feed.t_min
    entry_hit_gt = None
    while n_step < _MAX_FEED_ITERS:
        tip = measured_tip(ctx)
        clear = ctx.gates.tip_clearance(tip, ctx.hole, ctx.board_n)
        pen = max(0.0, -clear)
        ctx.st["depth"] = max(float(ctx.st["depth"]), pen)
        if entry_hit_gt is None and pen >= 0.001:
            # First millimetre into the face — true entry before the bit can twist.
            entry_hit_gt = _capture_entry_hit(ctx)
        if pen >= DRILL_DEPTH - 0.4 * DEPTH_TOL:
            ctx.log(f"depth reached: estimated {pen*1000:.1f} mm after {n_step} feed steps")
            break
        force_n = float(ctx.st.get("cut_force_now") or 0.0)
        raw_rpm = ctx.st.get("chuck_rpm")
        rpm = None if raw_rpm is None else float(raw_rpm)
        step, speed = ctrl.command(
            pen, max_along - along, dt, force_n=force_n, rpm=rpm,
            motor_on=bool(ctx.st["motor"]))
        if step <= 1e-5:
            if pen >= DRILL_DEPTH - DEPTH_TOL:
                break
            raise GateError(
                f"feed budget exhausted at estimated depth {pen*1000:.1f} mm "
                f"(target {DRILL_DEPTH*1000:.0f} mm; check FEED_MARGIN / tip)")
        along += step
        T = ctx.gates.feed_duration(step, max(feed.t_min, step / max(speed, 1e-6)))
        ctx.robot.move_along(ctx.d, step, ctx.Rd, T=T, speed=speed, settle=feed.settle)
        dt = T
        # Force / dry-cut gates run in tick callbacks; re-check estimated tip here too.
        if not ctx.st["motor"]:
            ctx.gates.check_dry_cut(estimated_penetration(ctx), False, "7/8 feed")
        n_step += 1
        if n_step == 1 or n_step % 3 == 0:
            scale = ctrl.load_scale(force_n, rpm, motor_on=bool(ctx.st["motor"]))
            rpm_s = "n/a" if rpm is None else f"{rpm:.0f}"
            ctx.log(f"  feed {n_step}: depth {pen*1000:.1f} mm, "
                    f"step {step*1000:.1f} mm @ {speed*1000:.0f} mm/s "
                    f"(F {force_n:.1f} N, {rpm_s} rpm, load*{scale:.2f})")
            ctx.runlog.event("7/8 pull trigger + drill", "feed",
                             along_mm=along * 1000.0, depth_mm=pen * 1000.0,
                             clear_mm=clear * 1000.0, speed_mm_s=speed * 1000.0,
                             step_mm=step * 1000.0, cut_force_n=force_n,
                             chuck_rpm=rpm, load_scale=scale)
    else:
        raise GateError(
            f"feed did not settle in {_MAX_FEED_ITERS} steps "
            f"(depth {ctx.st['depth']*1000:.1f} mm, target {DRILL_DEPTH*1000:.0f} mm)")
    ctx.st["entry_hit_gt"] = entry_hit_gt

    depth = estimated_penetration(ctx, ctx.hole, ctx.board_n)
    ctx.st["depth"] = max(float(ctx.st["depth"]), depth)
    if depth < DRILL_DEPTH - DEPTH_TOL:
        raise GateError(
            f"estimated depth only {depth*1000:.1f} mm, need {DRILL_DEPTH*1000:.0f} mm")
    if depth > DRILL_DEPTH + FEED_MARGIN + DEPTH_TOL:
        raise GateError(
            f"estimated depth {depth*1000:.1f} mm overshot target "
            f"{DRILL_DEPTH*1000:.0f} mm by more than {FEED_MARGIN*1000:.0f} mm margin")

    ctx.log(f"hole depth reached {depth*1000:.1f} mm (estimated tip vs detected plane)")
    log_tip_vs_gt(ctx, "post-feed")
    if ctx.st.get("cut_force_n"):
        ctx.log(f"cutting: peak thrust {ctx.st['cut_force_n']:.1f} N, "
                f"peak torque {ctx.st['cut_torque_nm']:.3f} N*m, "
                f"chuck {ctx.st.get('chuck_rpm', 0):.0f} rpm")
        ctx.metrics["cut_force_n"] = round(float(ctx.st["cut_force_n"]), 2)
        ctx.metrics["cut_torque_nm"] = round(float(ctx.st["cut_torque_nm"]), 4)
        ctx.metrics["chuck_rpm"] = round(float(ctx.st.get("chuck_rpm") or 0.0), 1)
    _record_accuracy(ctx)
    retract = ctx.motion["retract"]
    p_now, _ = ctx.robot.site_pose()
    back = float((ctx.p_aim - p_now) @ ctx.d)
    if abs(back) > 1e-4:
        kw = retract.cart_kwargs(abs(back))
        ctx.robot.move_along(ctx.d, back, ctx.Rd, T=kw["T"], speed=retract.vmax, settle=kw["settle"])
    ctx.hand.set_angles(overlay_index(wrap, 1000), speed=500); ctx.robot.wait(1.0)
    kw = retract.cart_kwargs(0.10)
    ctx.robot.move_along(-ctx.d, 0.10, ctx.Rd, T=kw["T"], speed=retract.vmax, settle=kw["settle"])
    kw = retract.cart_kwargs(0.06)
    ctx.robot.move_to_pose(ctx.robot.site_pose()[0] + 0.06 * UP, ctx.Rd, **kw)
    ctx.phase_end("7/8 pull trigger + drill",
                  depth_mm=depth * 1000.0,
                  depth_err_mm=(depth - DRILL_DEPTH) * 1000.0,
                  entry_err_vs_gt_mm=ctx.metrics.get("entry_err_vs_gt_mm"),
                  feed_steps=n_step, feed_along_mm=along * 1000.0,
                  cut_force_n=ctx.metrics.get("cut_force_n"),
                  cut_torque_nm=ctx.metrics.get("cut_torque_nm"),
                  chuck_rpm=ctx.metrics.get("chuck_rpm"))


def _gt_penetration(ctx):
    """Eval-only: MuJoCo bit_tip site vs the GT board face. None on --backend real."""
    if ctx.data is None or ctx.tip_site is None or int(ctx.tip_site) < 0:
        return None, None
    tip = ctx.data.site_xpos[ctx.tip_site].copy()
    n_gt = ctx.eval_normal()
    return float((tip - ctx.eval_hole()) @ n_gt), tip


def _pierce_along(tip, bit_dir, hole, normal):
    """Intersect the bit ray through `tip` with the face plane."""
    n = np.asarray(normal, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    d = np.asarray(bit_dir, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    dn = float(d @ n)
    if abs(dn) < 1e-6:
        return np.asarray(tip, float) - float((np.asarray(tip) - hole) @ n) * n
    s = float((np.asarray(hole, float) - np.asarray(tip, float)) @ n) / dn
    return np.asarray(tip, float) + s * d


def _capture_entry_hit(ctx):
    """World pierce point of the real tip at first contact (sim eval)."""
    pen, tip = _gt_penetration(ctx)
    if tip is None:
        return None
    if ctx.data is None:
        return _pierce_along(tip, ctx.d, ctx.eval_hole(), ctx.eval_normal())
    bit = ctx.data.body("drill").xmat.reshape(3, 3)[:, 0]
    return _pierce_along(tip, bit, ctx.eval_hole(), ctx.eval_normal())


def _log_gt_planar_at_aim(ctx):
    """Where the real tip would pierce, right before feed (eval-only)."""
    pen, tip = _gt_penetration(ctx)
    if tip is None:
        return
    n = ctx.eval_normal()
    hit = _pierce_along(tip, ctx.d, ctx.eval_hole(), n)
    err, ey, ez = planar_err_mm(hit, ctx.eval_hole(), n)
    err_m, _, _ = planar_err_mm(hit, ctx.hole, ctx.board_n)
    clear = float((ctx.eval_hole() - tip) @ n)
    ctx.log(f"  GT tip planar at aim: vs GT hole {err:.1f} mm (Y {ey:+.1f}, Z {ez:+.1f}), "
            f"vs marker {err_m:.1f} mm, clear {clear*1000:.1f} mm")


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
    n_gt = ctx.eval_normal()
    Rdr = ctx.data.body("drill").xmat.reshape(3, 3)
    bit_actual = Rdr[:, 0]
    # Prefer the pierce captured at first penetration (before cut-twist); fall
    # back to projecting the final tip along the current bit.
    hit_gt = ctx.st.get("entry_hit_gt")
    if hit_gt is None:
        hit_gt = _pierce_along(tip_gt, bit_actual, ctx.eval_hole(), n_gt)
    err_gt, ey, ez = planar_err_mm(hit_gt, ctx.eval_hole(), n_gt)
    err_cmd, _, _ = planar_err_mm(hit_gt, ctx.hole, ctx.board_n)
    err_mark = err_cmd
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
