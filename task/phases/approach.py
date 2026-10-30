"""Phase 6: visual servo to the mark — planar (Y/Z) + tip-to-face depth."""
import numpy as np
from safety import GateError

from task.constants import (
    AIM_STANDOFF, APPROACH_CLEAR, CLEAR_TOL, DRILL_DEPTH, FEED_MARGIN,
    LAT_TOL, MARKER_ALPHA, MARKER_MIN_N_DOT, MARKER_N_ALPHA, MAX_APPROACH_STEP,
    MAX_HOLE_JUMP,
)
from task.geom import aim_rotation, bit_direction
from task.held import relocate_held_drill
from task.replan import ReplanConfig, should_replan
from task.scene_est import board_usable, update_scene_from_ctx
from task.state_machine import TaskState, TaskStateMachine
from task.target_filter import TargetFilter
from task.tip import log_tip_vs_gt, measured_tip, rebind_held_tip


def _ensure_filter(ctx) -> TargetFilter:
    filt = getattr(ctx, "target_filter", None)
    if filt is None:
        filt = TargetFilter(
            alpha=MARKER_ALPHA, n_alpha=MARKER_N_ALPHA,
            max_jump=MAX_HOLE_JUMP, min_n_dot=MARKER_MIN_N_DOT)
        if ctx.hole is not None:
            filt.seed(ctx.hole, ctx.board_n)
        ctx.target_filter = filt
    return filt


def approach_board(ctx):
    ctx.phase("6/8 approach by tip-to-board distance")
    sm = getattr(ctx, "state_machine", None) or TaskStateMachine()
    ctx.state_machine = sm
    replan_cfg = ReplanConfig()
    scene_prev = getattr(ctx, "scene", None)
    scene = update_scene_from_ctx(ctx)
    ok, why = board_usable(scene)
    if not ok:
        sm.abort(why)
        raise GateError(f"refuse board approach: {why}")
    decision = should_replan(
        scene_prev=scene_prev, scene_now=scene, cfg=replan_cfg,
        robot_moving=False,
        retries_used=int(sm.retries.get("approach", 0)))
    if decision.needed:
        ctx.log(f"replan trigger: {decision.reason}")
        ctx.runlog.event("6/8 approach", "replan", reason=decision.reason,
                         can_retry=decision.can_retry)
        ctx.metrics["replan_count"] = int(ctx.metrics.get("replan_count") or 0) + 1
        if not decision.can_retry:
            sm.abort(decision.reason)
            raise GateError(f"replan exhausted: {decision.reason}")
        sm.bump_retry("approach", replan_cfg.max_retries)
        # Re-validate board from wrist before transferring (no stale GT fallback).
        from task.phases.board import locate_board
        locate_board(ctx)
        scene = update_scene_from_ctx(ctx)
        ok, why = board_usable(scene)
        if not ok:
            sm.abort(why)
            raise GateError(f"refuse board approach after replan: {why}")
    sm.enter(TaskState.RETREAT_OR_REPOSITION, reason="held transfer")
    _ensure_filter(ctx)
    d, board_n, hole, Rd = ctx.d, ctx.board_n, ctx.hole, ctx.Rd
    dn = max(float(d @ board_n), ctx.gates.MIN_FEED_DOT)
    p_far = hole - Rd @ ctx.tip_in_hand - (APPROACH_CLEAR / dn) * d
    ctx.gates.check_workspace(p_far, "approach-far")
    ctx.gates.check_aim_clearance(p_far, Rd, ctx.tip_in_hand, hole, board_n, along_d=0.0)
    relocate_held_drill(
        ctx, p_far, Rd, profile="relocate", name="approach-far",
        final_profile="approach")
    ctx.robot.wait(0.2)
    sm.enter(TaskState.APPROACH_BOARD, reason="tip-to-face")
    _second_look(ctx)
    # Drill may have seated during the reorient; refresh lateral tip offset.
    rebind_held_tip(ctx, "approach-far")
    d, board_n, hole, Rd = ctx.d, ctx.board_n, ctx.hole, ctx.Rd

    def tip_world():
        return measured_tip(ctx)

    def refresh_hole():
        """Re-see the mark; RGB ray-plane, so it still works when depth is invalid."""
        if ctx.args.board_target == "offset":
            return False
        try:
            capture = getattr(ctx.cam, "settle_capture", None) or ctx.cam.capture
            rgb, dep = capture()
            new, src, px = ctx.board_det.locate_target(
                rgb, dep, ctx.cam.pose(), ctx.board.point, ctx.board_n)
        except (RuntimeError, AttributeError):
            return False
        filt = _ensure_filter(ctx)
        hole_f, jump, ok = filt.update_hole(
            new, plane_point=ctx.board.point, plane_n=ctx.board_n)
        if not ok:
            ctx.log(f"  marker jump {jump*1000:.0f} mm ignored")
            ctx.runlog.event("6/8 approach by tip-to-board distance", "marker_jump",
                             jump_mm=jump * 1000.0)
            return False
        ctx.hole = hole_f
        ctx.st["hole"] = hole_f
        ctx.board.hole = hole_f
        ctx.board.target_px = px
        ctx.board.target_source = src
        return True

    clear = ctx.gates.tip_clearance(tip_world(), hole, board_n)
    ctx.log(f"approach start: tip-to-face {clear*1000:.1f} mm (target {AIM_STANDOFF*1000:.0f} mm)")
    refresh_hole()
    hole = ctx.hole
    lat0 = _planar_miss(tip_world(), d, hole, board_n, ctx.gates.MIN_FEED_DOT)
    ctx.runlog.event("6/8 approach by tip-to-board distance", "clearance",
                     clear_mm=clear * 1000.0, target_mm=AIM_STANDOFF * 1000.0,
                     lat_mm=lat0 * 1000.0)

    n_refresh = 0
    for k in range(50):
        if k % 2 == 0 and refresh_hole():
            n_refresh += 1
            hole = ctx.hole
        dn = max(float(d @ board_n), ctx.gates.MIN_FEED_DOT)
        tip = tip_world()
        tip_des = hole - (AIM_STANDOFF / dn) * d
        err = tip_des - tip
        clear = ctx.gates.tip_clearance(tip, hole, board_n)
        lat = _planar_miss(tip, d, hole, board_n, ctx.gates.MIN_FEED_DOT)
        if abs(clear - AIM_STANDOFF) <= CLEAR_TOL and lat <= LAT_TOL:
            ctx.log(f"approach settled: tip-to-face {clear*1000:.1f} mm, "
                    f"planar {lat*1000:.1f} mm after {k} corrections ({n_refresh} marker updates)")
            ctx.metrics["approach_clear_mm"] = round(float(clear) * 1000.0, 2)
            ctx.metrics["approach_lat_mm"] = round(float(lat) * 1000.0, 2)
            log_tip_vs_gt(ctx, "approach-settled")
            break
        nerr = float(np.linalg.norm(err))
        if nerr < 1e-6:
            continue
        step = err * min(1.0, MAX_APPROACH_STEP / nerr)
        step = clip_into_board_step(
            step, board_n, clear, AIM_STANDOFF - CLEAR_TOL)
        if float(np.linalg.norm(step)) < 1e-6:
            continue
        p_now, _ = ctx.robot.site_pose()
        p_next = p_now + step
        ctx.gates.check_workspace(p_next, "approach")
        ctx.gates.check_aim_clearance(p_next, Rd, ctx.tip_in_hand, hole, board_n, along_d=0.0)
        step_len = float(np.linalg.norm(step))
        kw = ctx.motion["approach"].cart_kwargs(step_len)
        ctx.robot.move_to_pose(p_next, Rd, **kw)
        ctx.gates.check_tracking(ctx.robot.max_track_err, "approach")
        ctx.gates.check_joint_margin(ctx.robot.limit_margin(ctx.robot.get_q()))
        if k % 5 == 0:
            ctx.log(f"  correct {k}: clear {clear*1000:.1f} mm, planar {lat*1000:.1f} mm "
                    f"-> step {np.linalg.norm(step)*1000:.1f} mm")
            ctx.runlog.event("6/8 approach by tip-to-board distance", "correct",
                             k=k, clear_mm=clear * 1000.0, lat_mm=lat * 1000.0,
                             step_mm=float(np.linalg.norm(step)) * 1000.0,
                             track_err_mm=float(ctx.robot.max_track_err) * 1000.0)
    else:
        raise GateError(
            f"approach failed to settle (clearance {clear*1000:.1f} mm, planar {lat*1000:.1f} mm)")

    p_aim, _ = ctx.robot.site_pose()
    ctx.gates.check_trigger_clearance(tip_world(), hole, board_n)
    feed = ctx.gates.feed_along_bit(AIM_STANDOFF, DRILL_DEPTH, FEED_MARGIN, d @ board_n)
    ctx.gates.check_aim_clearance(p_aim, Rd, ctx.tip_in_hand, hole, board_n, along_d=0.0)
    ctx.gates.check_aim_clearance(p_aim, Rd, ctx.tip_in_hand, hole, board_n, along_d=feed)
    ctx.p_aim, ctx.feed, ctx.hole = p_aim, feed, hole
    ctx.st["hole"] = hole
    ctx.phase_end("6/8 approach by tip-to-board distance",
                  clear_mm=float(clear) * 1000.0, lat_mm=float(lat) * 1000.0,
                  feed_mm=feed * 1000.0, marker_updates=n_refresh)


def nudge_to_aim(ctx, name="aim-nudge"):
    """Cartesian hops so the current estimated tip matches AIM_STANDOFF / hole."""
    d, board_n, hole, Rd = ctx.d, ctx.board_n, ctx.hole, ctx.Rd
    for _ in range(4):
        dn = max(float(d @ board_n), ctx.gates.MIN_FEED_DOT)
        tip = measured_tip(ctx)
        tip_des = hole - (AIM_STANDOFF / dn) * d
        err = tip_des - tip
        clear = ctx.gates.tip_clearance(tip, hole, board_n)
        lat = _planar_miss(tip, d, hole, board_n, ctx.gates.MIN_FEED_DOT)
        if abs(clear - AIM_STANDOFF) <= CLEAR_TOL and lat <= LAT_TOL:
            return
        nerr = float(np.linalg.norm(err))
        if nerr < 1e-6:
            return
        step = err * min(1.0, MAX_APPROACH_STEP / nerr)
        step = clip_into_board_step(step, board_n, clear, AIM_STANDOFF - CLEAR_TOL)
        if float(np.linalg.norm(step)) < 1e-6:
            return
        p_now, _ = ctx.robot.site_pose()
        p_next = p_now + step
        ctx.gates.check_workspace(p_next, name)
        ctx.gates.check_aim_clearance(p_next, Rd, ctx.tip_in_hand, hole, board_n, along_d=0.0)
        kw = ctx.motion["approach"].cart_kwargs(float(np.linalg.norm(step)))
        ctx.robot.move_to_pose(p_next, Rd, **kw)
        ctx.gates.check_tracking(ctx.robot.max_track_err, name)
        ctx.log(f"{name}: planar {lat*1000:.1f} mm, clear {clear*1000:.1f} mm "
                f"-> step {np.linalg.norm(step)*1000:.1f} mm")


def clip_into_board_step(step, board_n, clear, min_clear):
    """Drop the into-board component that would put the tip closer than `min_clear`."""
    n = np.asarray(board_n, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    step = np.asarray(step, float).copy()
    into = float(step @ n)
    cap = max(float(clear) - float(min_clear), 0.0)
    if into > cap:
        step = step - (into - cap) * n
    return step


def _planar_miss(tip, d, hole, board_n, min_dot):
    """Distance in the board plane between the bit's pierce point and `hole`."""
    n = np.asarray(board_n, float); n = n / (np.linalg.norm(n) + 1e-12)
    d = np.asarray(d, float); d = d / (np.linalg.norm(d) + 1e-12)
    dn = max(float(d @ n), min_dot)
    clear = float((np.asarray(hole) - np.asarray(tip)) @ n)
    hit = np.asarray(tip, float) + (clear / dn) * d
    lat = hit - np.asarray(hole, float)
    lat = lat - n * float(lat @ n)
    return float(np.linalg.norm(lat))


def _second_look(ctx):
    """Closer RGB-D pass (~approach-far) to refresh the plane and mark."""
    try:
        capture = getattr(ctx.cam, "settle_capture", None) or ctx.cam.capture
        rgb, dep = capture()
        search = ctx.board.point if ctx.board is not None else ctx.calib.board.position
        board = ctx.board_det.detect(
            rgb, dep, ctx.cam.pose(), normal_hint=ctx.board_n,
            search_center=search)
        ctx.gates.check_board(board, facing_hint=ctx.board_n)
        if board.target_source in ("marker", "aruco"):
            ctx.gates.check_hole_on_face(
                board.hole, board.point, board.normal, ctx.calib.board.half,
                name="second-look hole")
            if getattr(ctx.args, "random", None) is None:
                ctx.gates.check_hole_prior(board.hole, ctx.hole_hint, "second-look hole")
        else:
            ctx.gates.check_hole_prior(board.hole, ctx.hole_hint, "second-look hole")
        filt = _ensure_filter(ctx)
        n_filt = filt.update_normal(board.normal)
        if n_filt is not None:
            d = bit_direction(ctx.args.tilt, ctx.args.azim, n_filt)
            Rd = aim_rotation(d, n_filt)
            ctx.gates.check_drill_angles(
                ctx.args.tilt, ctx.args.azim, d, n_filt,
                allow_angled=ctx.allow_angled)
            ctx.board_n = n_filt
            ctx.st["board_n"] = n_filt
            ctx.d = d
            ctx.Rd = Rd
            ctx.st["drill_dir"] = d
        hole_f, jump, ok = filt.update_hole(
            board.hole, plane_point=board.point, plane_n=ctx.board_n)
        if not ok:
            ctx.log(f"second look hole jump {jump*1000:.0f} mm — keep far detection")
            ctx.runlog.event("6/8 approach by tip-to-board distance", "second_look_jump",
                             jump_mm=jump * 1000.0)
            return
        board.hole = hole_f
        board.normal = np.asarray(ctx.board_n, float).copy()
        ctx.board = board
        ctx.hole = hole_f
        ctx.st["hole"] = ctx.hole
        err_h = float(np.linalg.norm(ctx.hole - ctx.eval_hole()) * 1000.0)
        ctx.metrics["board_hole_err_mm"] = round(err_h, 2)
        ctx.metrics["target_source"] = board.target_source
        ctx.show_detection(ctx.board_det.draw(rgb, board, ctx.cam.pose()))
        ctx.log(f"second look: target via {board.target_source}, hole vs prior {err_h:.1f} mm")
        ctx.runlog.event("6/8 approach by tip-to-board distance", "second_look",
                         hole_err_mm=err_h, target_source=board.target_source,
                         jump_mm=jump * 1000.0, filter_updates=filt.n_updates)
    except GateError:
        raise
    except RuntimeError as e:
        ctx.log(f"second look failed ({e}); keep far detection")
        ctx.runlog.event("6/8 approach by tip-to-board distance", "second_look_fail", reason=str(e))
