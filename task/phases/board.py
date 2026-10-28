"""Phase 5: locate the board with the wrist camera after grasp."""
import numpy as np

from task.constants import UP
from task.geom import aim_rotation, bit_direction
from task.held import relocate_held_drill


def locate_board(ctx):
    ctx.phase("5/8 locate board")
    look = ctx.hole_hint.copy()
    cam_guess = look - 0.38 * ctx.board_n_hint + np.array([0.0, 0.0, 0.08])
    p_view, R_view = ctx.views.site_target_for_view(cam_guess, look, up_hint=UP)
    ctx.gates.check_workspace(p_view, "board-view")
    relocate_held_drill(ctx, p_view, R_view, T=2.5, name="board-view")
    ctx.robot.wait(0.4)
    # Drop mid-motion frames from the latency buffer before looking at the board.
    capture = getattr(ctx.cam, "settle_capture", None) or ctx.cam.capture
    rgb, dep = capture()
    board = ctx.board_det.detect(
        rgb, dep, ctx.cam.pose(), normal_hint=ctx.board_n_hint,
        search_center=ctx.calib.board.position)
    ctx.gates.check_board(board, facing_hint=ctx.board_n_hint)
    ctx.gates.check_hole_prior(board.hole, ctx.hole_hint, "far-look hole")
    ctx.show_detection(ctx.board_det.draw(rgb, board, ctx.cam.pose()))
    err_n = float(np.degrees(np.arccos(np.clip(board.normal @ ctx.board_n_hint, -1, 1))))
    err_h = float(np.linalg.norm(board.hole - ctx.hole_hint) * 1000)
    ctx.log(f"board: n=({board.normal[0]:+.3f},{board.normal[1]:+.3f},{board.normal[2]:+.3f}), "
            f"hole=({board.hole[0]:.3f},{board.hole[1]:.3f},{board.hole[2]:.3f}), "
            f"pts={board.n_points}, inliers={board.inlier_ratio:.0%}")
    ctx.log(f"   vs nominal: normal err {err_n:.1f} deg, hole err {err_h:.1f} mm")
    ctx.metrics["board_normal_err_deg"] = round(err_n, 2)
    ctx.metrics["board_hole_err_mm"] = round(err_h, 2)
    ctx.metrics["target_source"] = board.target_source

    board_n = board.normal
    hole = board.hole.copy()
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
                  inliers=board.inlier_ratio)
