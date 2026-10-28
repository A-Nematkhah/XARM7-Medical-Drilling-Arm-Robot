"""Phase 6: visual servo to the mark — planar (Y/Z) + tip-to-face depth."""
import numpy as np
from safety import GateError

from task.constants import (
    AIM_STANDOFF, APPROACH_CLEAR, CLEAR_TOL, DRILL_DEPTH, FEED_MARGIN,
    LAT_TOL, MAX_APPROACH_STEP, MAX_HOLE_JUMP,
)
from task.geom import aim_rotation, bit_direction
from task.held import relocate_held_drill
from task.tip import measured_tip


def approach_board(ctx):
    ctx.phase("6/8 approach by tip-to-board distance")
    d, board_n, hole, Rd = ctx.d, ctx.board_n, ctx.hole, ctx.Rd
    dn = max(float(d @ board_n), ctx.gates.MIN_FEED_DOT)
    p_far = hole - Rd @ ctx.tip_in_hand - (APPROACH_CLEAR / dn) * d
    ctx.gates.check_workspace(p_far, "approach-far")
    ctx.gates.check_aim_clearance(p_far, Rd, ctx.tip_in_hand, hole, board_n, along_d=0.0)
    relocate_held_drill(ctx, p_far, Rd, T=2.0, name="approach-far")
    ctx.robot.wait(0.2)
    _second_look(ctx)
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
        jump = float(np.linalg.norm(new - ctx.hole))
        if jump > MAX_HOLE_JUMP:
            ctx.log(f"  marker jump {jump*1000:.0f} mm ignored")
            ctx.runlog.event("6/8 approach by tip-to-board distance", "marker_jump",
                             jump_mm=jump * 1000.0)
            return False
        ctx.hole = new
        ctx.st["hole"] = new
        ctx.board.hole = new
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
            break
        nerr = float(np.linalg.norm(err))
        if nerr < 1e-6:
            continue
        step = err * min(1.0, MAX_APPROACH_STEP / nerr)
        p_now, _ = ctx.robot.site_pose()
        p_next = p_now + step
        ctx.gates.check_workspace(p_next, "approach")
        ctx.gates.check_aim_clearance(p_next, Rd, ctx.tip_in_hand, hole, board_n, along_d=0.0)
        ctx.robot.move_to_pose(p_next, Rd, T=max(0.35, float(np.linalg.norm(step)) / 0.04))
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
        board = ctx.board_det.detect(
            rgb, dep, ctx.cam.pose(), normal_hint=ctx.board_n,
            search_center=ctx.calib.board.position)
        ctx.gates.check_board(board, facing_hint=ctx.board_n)
        jump = float(np.linalg.norm(board.hole - ctx.hole))
        if jump > MAX_HOLE_JUMP:
            ctx.log(f"second look hole jump {jump*1000:.0f} mm — keep far detection")
            ctx.runlog.event("6/8 approach by tip-to-board distance", "second_look_jump",
                             jump_mm=jump * 1000.0)
            return
        ctx.gates.check_hole_prior(board.hole, ctx.hole_hint, "second-look hole")
        if float(board.normal @ ctx.board_n) > 0.97:
            ctx.board_n = board.normal
            ctx.st["board_n"] = board.normal
            ctx.d = bit_direction(ctx.args.tilt, ctx.args.azim, ctx.board_n)
            ctx.Rd = aim_rotation(ctx.d, ctx.board_n)
            ctx.st["drill_dir"] = ctx.d
        ctx.board = board
        ctx.hole = board.hole.copy()
        ctx.st["hole"] = ctx.hole
        err_h = float(np.linalg.norm(ctx.hole - ctx.hole_hint) * 1000.0)
        ctx.metrics["board_hole_err_mm"] = round(err_h, 2)
        ctx.metrics["target_source"] = board.target_source
        ctx.show_detection(ctx.board_det.draw(rgb, board, ctx.cam.pose()))
        ctx.log(f"second look: target via {board.target_source}, hole vs prior {err_h:.1f} mm")
        ctx.runlog.event("6/8 approach by tip-to-board distance", "second_look",
                         hole_err_mm=err_h, target_source=board.target_source)
    except RuntimeError as e:
        ctx.log(f"second look failed ({e}); keep far detection")
        ctx.runlog.event("6/8 approach by tip-to-board distance", "second_look_fail", reason=str(e))
