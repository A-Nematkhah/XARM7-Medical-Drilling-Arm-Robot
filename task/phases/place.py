"""Phase 8: put the drill back and retreat."""
import numpy as np

from task.constants import STANDOFF
from task.held import relocate_held_drill


def place_drill(ctx):
    if ctx.args.no_place:
        return
    ctx.phase("8/8 place the drill back")
    pg, Rg, a, p_slide = ctx.pg, ctx.Rg, ctx.a, ctx.p_slide
    relocate_held_drill(ctx, pg + np.array([0.0, 0.0, 0.12]), Rg, T=2.5, name="return")
    p_place = pg + [0, 0, 0.004]
    wps = [pg + [0, 0, 0.12], pg + [0, 0, 0.06], p_place, pg - STANDOFF * a, p_slide]
    chain, margin = ctx.robot.plan_path_ik(wps, Rg, min_margin=0.1)
    ctx.gates.check_joint_margin(margin)
    ctx.robot.move_joint_path(chain, T_first=3.0, T_step=0.8)
    ctx.gates.check_joint_margin(ctx.robot.limit_margin(ctx.robot.get_q()))
    ctx.robot.move_to_pose(p_place, Rg, T=2.0); ctx.robot.wait(0.3)
    ctx.log(f"place: max tracking err {ctx.robot.max_track_err*1000:.1f} mm")

    def tilt():
        if ctx.data is None:
            return 0.0
        return np.degrees(np.arccos(np.clip(ctx.data.body("drill").xmat.reshape(3, 3)[2, 2], -1, 1)))

    ctx.hand.set_angles([1000, 1000, 1000, 1000, 1000, 0], speed=600); ctx.robot.wait(1.2)
    ctx.log(f"  released, tilt {tilt():.1f}")
    ctx.robot.move_to_pose(pg - STANDOFF * a, Rg, T=0.6)
    ctx.robot.move_to_pose(p_slide, Rg, T=1.8)
    ctx.log(f"  slid out, tilt {tilt():.1f}, max tracking {ctx.robot.max_track_err*1000:.1f} mm")
    ctx.robot.move_to_pose(p_slide - 0.09 * a, Rg, T=1.2)
    ctx.robot.move_to_pose(p_slide - 0.09 * a + [0, 0, 0.12], Rg, T=1.2)
    ctx.log(f"  backed off, tilt {tilt():.1f}, max tracking {ctx.robot.max_track_err*1000:.1f} mm")
    ctx.hand.set_angles(ctx.hand.OPEN, speed=800)
    ctx.robot.move_joints(ctx.robot.rest_q, T=2.5)
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
