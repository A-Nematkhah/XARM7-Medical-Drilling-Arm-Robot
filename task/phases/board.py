"""Phase 5: locate the board. LiDAR points the wrist; D435i finds the mark."""
import numpy as np
from safety import GateError

from task.constants import MARKER_ALPHA, MARKER_MIN_N_DOT, MARKER_N_ALPHA, MAX_HOLE_JUMP, UP
from task.geom import aim_rotation, bit_direction
from task.held import relocate_held_drill
from task.scene_est import board_usable, update_scene_from_ctx
from task.state_machine import TaskState, TaskStateMachine
from task.target_filter import TargetFilter


def locate_board(ctx):
    ctx.phase("5/8 locate board")
    sm = getattr(ctx, "state_machine", None) or TaskStateMachine()
    ctx.state_machine = sm
    sm.enter(TaskState.RELOCALIZE_BOARD, reason="post-grasp")
    # Stale calib / spawn pose must not be used as a live pierce target.
    look, n_hint, search = _lidar_board_view(ctx)
    board = _from_wrist_look(ctx, look, n_hint, search)
    _commit_board(ctx, board, "lidar+wrist" if getattr(ctx, "coarse_map", None) else "wrist")
    scene = update_scene_from_ctx(ctx)
    ok, why = board_usable(scene)
    ctx.metrics["board_reloc_ok"] = 1 if ok else 0
    if not ok:
        sm.abort(why)
        ctx.runlog.event("5/8 locate board", "board_unusable", reason=why, **scene.to_log())
        raise GateError(f"board relocalization unusable: {why}")
    sm.enter(TaskState.PLAN_TRANSFER, reason="board locked")


def _lidar_board_view(ctx):
    """Wrist view target from the 360° hand scan (cached at grasp)."""
    cmap = getattr(ctx, "coarse_map", None)
    lidar = getattr(ctx, "lidar", None)
    if cmap is None and lidar is not None:
        try:
            cmap = lidar.locate()
            ctx.coarse_map = cmap
        except Exception as e:
            ctx.log(f"lidar board rescan miss ({e})")
            cmap = None
    if cmap is not None and cmap.board_xy is not None:
        z = float(ctx.hole_hint[2]) if ctx.hole_hint is not None else 0.34
        look = np.array([cmap.board_xy[0], cmap.board_xy[1], z])
        n = cmap.board_n
        if n is None:
            n = look.copy(); n[2] = 0.0
            n = n / (np.linalg.norm(n) + 1e-12)
        n = np.asarray(n, float)
        n = n / (np.linalg.norm(n) + 1e-12)
        ctx.log(f"wrist look from lidar board ({look[0]:.3f}, {look[1]:.3f})")
        return look, n, look
    return ctx.hole_hint.copy(), ctx.board_n_hint.copy(), ctx.calib.board.position


def _from_wrist_look(ctx, look, n_hint, search):
    cam_guess = look - 0.38 * n_hint + np.array([0.0, 0.0, 0.08])
    p_view, R_view = ctx.views.site_target_for_view(cam_guess, look, up_hint=UP)
    ctx.gates.check_workspace(p_view, "board-view")
    relocate_held_drill(
        ctx, p_view, R_view, profile="relocate", name="board-view",
        final_profile="approach")
    ctx.robot.wait(0.4)
    capture = getattr(ctx.cam, "settle_capture", None) or ctx.cam.capture
    rgb, dep = capture()
    board = ctx.board_det.detect(
        rgb, dep, ctx.cam.pose(), normal_hint=n_hint, search_center=search)
    ctx.gates.check_board(board, facing_hint=n_hint)
    _gate_hole(ctx, board, "far-look hole")
    ctx.show_detection(ctx.board_det.draw(rgb, board, ctx.cam.pose()))
    return board


def _gate_hole(ctx, board, name):
    if board.target_source in ("marker", "aruco"):
        ctx.gates.check_hole_on_face(
            board.hole, board.point, board.normal, ctx.calib.board.half, name=name)
        if getattr(ctx.args, "random", None) is None:
            ctx.gates.check_hole_prior(board.hole, ctx.hole_hint, name)
        return
    ctx.gates.check_hole_prior(board.hole, ctx.hole_hint, name)


def _commit_board(ctx, board, locate_src):
    n_ref = ctx.eval_normal()
    hole_ref = ctx.eval_hole()
    err_n = float(np.degrees(np.arccos(np.clip(board.normal @ n_ref, -1, 1))))
    err_h = float(np.linalg.norm(board.hole - hole_ref) * 1000)
    ctx.log(f"board: n=({board.normal[0]:+.3f},{board.normal[1]:+.3f},{board.normal[2]:+.3f}), "
            f"hole=({board.hole[0]:.3f},{board.hole[1]:.3f},{board.hole[2]:.3f}), "
            f"pts={board.n_points}, inliers={board.inlier_ratio:.0%}")
    ctx.log(f"   vs spawn GT: normal err {err_n:.1f} deg, hole err {err_h:.1f} mm "
            f"(locate={locate_src})")
    ctx.metrics["board_normal_err_deg"] = round(err_n, 2)
    ctx.metrics["board_hole_err_mm"] = round(err_h, 2)
    ctx.metrics["target_source"] = board.target_source
    ctx.metrics["locate_source"] = locate_src

    board_n = board.normal
    hole = board.hole.copy()
    filt = TargetFilter(
        alpha=MARKER_ALPHA, n_alpha=MARKER_N_ALPHA,
        max_jump=MAX_HOLE_JUMP, min_n_dot=MARKER_MIN_N_DOT)
    filt.seed(hole, board_n)
    ctx.target_filter = filt
    d = bit_direction(ctx.args.tilt, ctx.args.azim, board_n)
    Rd = aim_rotation(d, board_n)
    ctx.gates.check_drill_angles(ctx.args.tilt, ctx.args.azim, d, board_n,
                                 allow_angled=ctx.allow_angled)
    ctx.st["board_n"], ctx.st["hole"], ctx.st["drill_dir"] = board_n, hole, d
    ctx.board, ctx.board_n, ctx.hole, ctx.d, ctx.Rd = board, board_n, hole, d, Rd
    ctx.log(f"board target via {board.target_source}")
    ctx.log(f"drill angle tilt={ctx.args.tilt:+.1f} deg, azim={ctx.args.azim:+.1f} deg, "
            f"dir=({d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f})")
    ctx.phase_end("5/8 locate board",
                  normal_err_deg=err_n, hole_err_mm=err_h,
                  target_source=board.target_source, n_points=board.n_points,
                  inliers=board.inlier_ratio, locate_source=locate_src)
