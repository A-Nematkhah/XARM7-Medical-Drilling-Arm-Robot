"""Phase 8: put the drill back and retreat."""
import numpy as np

from task.constants import STANDOFF
from task.held import relocate_held_drill
from task.motion_ctrl import safe_goto_joints


def place_drill(ctx):
    if ctx.args.no_place:
        return
    ctx.phase("8/8 place the drill back")
    place = ctx.motion["place"]
    pg, Rg, a, p_slide = ctx.pg, ctx.Rg, ctx.a, ctx.p_slide
    hover = pg + np.array([0.0, 0.0, 0.12])
    relocate_held_drill(ctx, hover, Rg, profile="relocate", name="return")
    p_mid = pg + np.array([0.0, 0.0, 0.06])
    p_place = pg + np.array([0.0, 0.0, 0.004])
    for p, tag, dist in ((p_mid, "place-mid", 0.06), (p_place, "place-down", 0.06)):
        ctx.gates.check_workspace(p, tag)
        ctx.robot.move_to_pose(p, Rg, **place.cart_kwargs(dist))
        ctx.gates.check_tracking(ctx.robot.max_track_err, tag)
        ctx.gates.check_joint_margin(ctx.robot.limit_margin(ctx.robot.get_q()))
    ctx.robot.wait(0.3)
    ctx.log(f"place: max tracking err {ctx.robot.max_track_err*1000:.1f} mm")

    def tilt():
        if ctx.data is None:
            return 0.0
        return np.degrees(np.arccos(np.clip(ctx.data.body("drill").xmat.reshape(3, 3)[2, 2], -1, 1)))

    ctx.hand.set_angles([1000, 1000, 1000, 1000, 1000, 0], speed=600); ctx.robot.wait(1.2)
    ctx.st["holding"] = False
    ctx.log(f"  released, tilt {tilt():.1f}")
    ctx.robot.move_to_pose(pg - STANDOFF * a, Rg, **place.cart_kwargs(STANDOFF))
    ctx.robot.move_to_pose(p_slide, Rg, **place.cart_kwargs(0.12))
    ctx.log(f"  slid out, tilt {tilt():.1f}, max tracking {ctx.robot.max_track_err*1000:.1f} mm")
    ctx.robot.move_to_pose(p_slide - 0.09 * a, Rg, **place.cart_kwargs(0.09))
    ctx.robot.move_to_pose(p_slide - 0.09 * a + [0, 0, 0.12], Rg, **place.cart_kwargs(0.12))
    ctx.log(f"  backed off, tilt {tilt():.1f}, max tracking {ctx.robot.max_track_err*1000:.1f} mm")
    ctx.hand.set_angles(ctx.hand.OPEN, speed=800)
    safe_goto_joints(
        ctx.robot, ctx.robot.rest_q, gates=ctx.gates, T=place.joint_T)
    ctx.log(f"  home, tilt {tilt():.1f}")
    if ctx.data is None:
        place_disp = 0.0
        up_axis = np.array([0.0, 0.0, 1.0])
    else:
        up_axis = ctx.data.body("drill").xmat.reshape(3, 3)[:, 2]
        place_disp = float(np.linalg.norm(ctx.data.body("drill").xpos[:2] - ctx.gt_grip) * 1000.0)
    ctx.log(f"drill back on the table, tilt {np.degrees(np.arccos(np.clip(up_axis[2], -1, 1))):.1f} deg, "
            f"displaced {place_disp:.0f} mm from start")
    ctx.metrics["place_disp_mm"] = round(place_disp, 1)
    ctx.phase_end("8/8 place the drill back", place_disp_mm=place_disp, tilt_deg=float(tilt()))
