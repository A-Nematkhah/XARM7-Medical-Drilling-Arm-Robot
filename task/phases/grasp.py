"""Phases 1–4: perceive the drill, grasp, lift, retreat the bit from the board."""
import os

import cv2
import numpy as np

from calibration import on_board_face
from safety import GateError
from task.constants import (
    GRIP_BELOW_HOUSING, HAND_GRASP, SAFE_TIP, SLIDE, STANDOFF, UP,
)
from task.tip import bind_tip_in_hand, measured_tip


def perceive_and_grasp(ctx):
    data, robot, cam, views, det, gates = (
        ctx.data, ctx.robot, ctx.cam, ctx.views, ctx.det, ctx.gates)
    capture = getattr(cam, "settle_capture", None) or cam.capture
    if data is not None:
        ctx.gt_grip = data.body("drill").xpos[:2].copy()
    else:
        ctx.gt_grip = np.asarray(ctx.xy, float)[:2].copy()
    ctx.log(f"drill placed at ({ctx.xy[0]:.3f}, {ctx.xy[1]:.3f}), "
            f"yaw {np.degrees(ctx.yaw):+.1f} deg (ground truth, for reporting only)")
    robot.wait(0.3)

    ctx.phase("1/8 look: top-down scan")
    ws = np.array([0.48, 0.0, 0.0])
    p, R = views.site_target_for_view(ws + [-0.05, 0, 0.60], ws, up_hint=(0, 1, 0))
    robot.goto_pose_joint_space(p, R, T=2.5); robot.wait(0.4)
    rgb, dep = capture()
    c_xy, ax = det.detect_coarse(rgb, dep, cam.pose())
    ctx.log(f"coarse: drill near ({c_xy[0]:.3f}, {c_xy[1]:.3f}), "
            f"axis yaw {np.degrees(np.arctan2(ax[1], ax[0])):+.1f} deg")
    ctx.metrics["coarse_xy_err_mm"] = round(float(np.linalg.norm(c_xy - ctx.gt_grip) * 1000.0), 2)
    ctx.show_detection(rgb, 1.5)
    ctx.phase_end("1/8 look: top-down scan", coarse_xy_err_mm=ctx.metrics["coarse_xy_err_mm"])

    ctx.phase("2/8 look: side view")
    side = np.cross(UP, [ax[0], ax[1], 0.0])
    look = np.array([c_xy[0], c_xy[1], 0.11])
    targets = [look + [0, 0, dz] for dz in (-0.03, 0.0, 0.04, 0.09)] + [
        look + 0.12 * np.array([ax[0], ax[1], 0.8])]
    q = None
    ax3 = np.array([ax[0], ax[1], 0.0])
    for dist, hgt in [(0.34, 0.25), (0.30, 0.22), (0.38, 0.28)]:
        for up in (UP, -UP, ax3, -ax3):
            p, R = views.site_target_for_view(look + dist * side + [0, 0, hgt], look, up_hint=up)
            for cand in robot.ik_solutions(p, R):
                if views.line_of_sight_clear(ctx.model, cand, targets, robot.qadr):
                    q = cand
                    break
            if q is not None:
                break
        if q is not None:
            break
    if q is None:
        raise RuntimeError("no reachable, unoccluded side view")
    robot.move_joints(q, T=2.5); robot.wait(0.4)
    rgb, dep = capture()
    os.makedirs(os.path.join(ctx.here, "outputs"), exist_ok=True)
    cv2.imwrite(os.path.join(ctx.here, "outputs", "side_view_raw.png"), rgb[:, :, ::-1])
    ctx.log(f"side view from ({cam.pose()[0][0]:.2f}, {cam.pose()[0][1]:.2f}, {cam.pose()[0][2]:.2f})")
    est = det.detect(rgb, dep, cam.pose(), axis_hint=ax)
    gates.check_perception(est, axis_hint=ax)
    ctx.show_detection(det.draw(rgb, est, cam.pose()))
    ctx.log(f"drill: grip ({est.grip_xy[0]:.4f}, {est.grip_xy[1]:.4f}) r={est.grip_radius*1000:.1f} mm, "
            f"yaw {np.degrees(est.yaw):+.1f} deg, housing underside z={est.housing_underside_z:.3f}")
    grip_err = float(np.linalg.norm(est.grip_xy - ctx.gt_grip) * 1000.0)
    yaw_err = float(np.degrees(np.angle(np.exp(1j * (est.yaw - ctx.yaw)))))
    ctx.log(f"   perception error vs ground truth: grip {grip_err:.1f} mm, yaw {yaw_err:+.1f} deg")
    ctx.metrics["grip_err_mm"] = round(grip_err, 2)
    ctx.metrics["drill_yaw_err_deg"] = round(yaw_err, 2)
    ctx.est = est
    ctx.phase_end("2/8 look: side view", grip_err_mm=grip_err, yaw_err_deg=yaw_err)

    a = est.axis / np.linalg.norm(est.axis)
    bz = np.cross(a, UP)
    Rg = np.column_stack([a, UP, bz])
    pg = np.array([*est.grip_xy, est.housing_underside_z - GRIP_BELOW_HOUSING])
    # Planned offset; rebound from the *actual* TCP after arriving at the grasp.
    ctx.tip_in_hand = bind_tip_in_hand(pg, Rg, est.bit_tip)
    ctx.a, ctx.Rg, ctx.pg = a, Rg, pg

    ctx.phase("3/8 approach (sideways, thumb pre-opposed)")
    ctx.hand.set_angles(ctx.hand.PRE_GRASP[:5] + [0], speed=800)
    options = [(SLIDE, 0.06, 0.10), (0.12, 0.04, 0.08), (0.12, 0.06, 0.05),
               (0.12, 0.01, 0.03), (0.12, 0.0, 0.02), (0.10, 0.03, 0.07)]
    gates.check_workspace(pg, "grasp")
    q_start = robot.get_q()
    last_err = None
    slide = margin = None
    p_slide = None
    arrived = False
    for side_sign in (1.0, -1.0):
        bz_s = side_sign * bz
        for slide, out, up in options:
            p_slide = pg - slide * bz_s - STANDOFF * a
            p_entry = p_slide - out * bz_s + [0, 0, up]
            wps = ([p_entry, p_slide]
                   + [pg - STANDOFF * a - s * slide * bz_s for s in (0.66, 0.33, 0.0)]
                   + [pg])
            try:
                chain, margin = robot.plan_path_ik(wps, Rg)
                gates.check_joint_margin(margin)
            except (RuntimeError, GateError) as ex:
                ctx.log(f"approach option side={side_sign:+.0f} {slide, out, up} rejected: {ex}")
                last_err = ex
                continue
            ctx.log(f"approach: side {side_sign:+.0f}, slide {slide*100:.0f} cm, "
                    f"joint-limit margin {margin:.2f} rad")
            try:
                # Stay on the solved IK branch; Cartesian slide from entry was
                # blowing a 90 mm tracking hole through the drill on tight yaws.
                robot.move_joint_path(chain[:-1], T_first=3.0, T_step=0.8)
                gates.check_joint_margin(robot.limit_margin(robot.get_q()))
                e = robot.move_to_pose(pg - STANDOFF * a, Rg, T=1.5)
                q_meas = robot.get_q()
                ctx.log(f"slide max tracking err {robot.max_track_err*1000:.1f} mm; "
                        f"residual {np.round(e[:3]*1000,1)} mm, "
                        f"{np.round(np.degrees(e[3:]),1)} deg; q={np.round(q_meas,2)}")
                gates.check_tracking(robot.max_track_err, "slide")
                gates.check_joint_margin(robot.limit_margin(q_meas))
                arrived = True
                break
            except GateError as ex:
                last_err = ex
                ctx.log(f"approach option side={side_sign:+.0f} {slide, out, up} "
                        f"failed in motion: {ex}")
                robot.move_joints(q_start, T=2.5)
        if arrived:
            break
    if not arrived:
        raise last_err if last_err is not None else RuntimeError(
            "drill is outside the reachable grasping workspace")
    ctx.p_slide = p_slide
    robot.move_to_pose(pg, Rg, T=0.6)
    p_now, R_now = robot.site_pose()
    ctx.tip_in_hand = bind_tip_in_hand(p_now, R_now, est.bit_tip)
    ctx.phase_end("3/8 approach (sideways, thumb pre-opposed)",
                  slide_cm=slide * 100.0, joint_margin=margin,
                  track_err_mm=float(robot.max_track_err) * 1000.0)

    ctx.phase("4/8 grasp")
    if data is not None:
        Rdr = data.body("drill").xmat.reshape(3, 3)
        ctx.log(f"drill disturbance during approach: "
                f"{np.linalg.norm(data.body('drill').xpos[:2]-ctx.gt_grip)*1000:.1f} mm, "
                f"{np.degrees(np.arctan2(Rdr[1,0], Rdr[0,0]) - ctx.yaw):+.1f} deg")
    ctx.hand.set_angles([0, 0, 0, 1000, 1000, 0], speed=800); robot.wait(1.0)
    ctx.hand.set_angles(HAND_GRASP, speed=800); robot.wait(1.0)
    p_now, R_now = robot.site_pose()
    t = bind_tip_in_hand(p_now, R_now, est.bit_tip)
    held = float(ctx.calib.tool.held_bit_reach)
    if held > 0:
        # Perception gives the bit direction in the grasp frame; the along-bit
        # length is the calibrated post-grasp TCP→tip (palm seating).
        t = np.asarray(t, float).copy()
        t[0] = np.sign(t[0] if abs(t[0]) > 1e-9 else 1.0) * held
    ctx.tip_in_hand = t
    ctx.log(f"tip_in_hand after grasp [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")
    f = ctx.hand.get_forces()
    ctx.log(f"finger torques [little ring middle index thumb_bend thumb_rot] = {np.round(f, 2)} N*m")
    gates.check_grasp(f)
    z0 = float(robot.site_pose()[0][2])
    robot.move_to_pose(pg + [0, 0, 0.15], Rg, T=1.8); robot.wait(0.3)
    lifted = float(robot.site_pose()[0][2]) - z0
    ctx.log(f"lifted TCP by {lifted*1000:.0f} mm (commanded 150)")
    gates.check_lift(lifted)

    p_hold, R_hold = robot.site_pose()
    bit = R_hold[:, 0]
    n = ctx.st["board_n"]
    hole = ctx.st["hole"]
    tip = measured_tip(ctx)
    tip_clear = gates.tip_clearance(tip, hole, n)
    hit = np.asarray(tip, float) + float(tip_clear) * np.asarray(n, float)
    on_face = on_board_face(hit, hole, n, ctx.calib.board.half)
    if tip_clear < SAFE_TIP and on_face:
        pull = -bit if float(bit @ n) > 0.25 else -n
        dn_pull = float(pull @ n)
        if abs(dn_pull) < 0.2:
            pull, dn_pull = -n, -1.0
        s = (tip_clear - SAFE_TIP) / dn_pull
        p_back = p_hold + s * pull
        p_back[2] = max(float(p_back[2]), float(p_hold[2]))
        g0 = gates.grasp_face_clearance(p_back, ctx.st["hole"], n)
        if g0 < gates.REORIENT_GRASP_STANDOFF:
            p_back = p_back - (gates.REORIENT_GRASP_STANDOFF - g0) * n
            p_back[2] = max(float(p_back[2]), float(p_hold[2]))
        gates.check_workspace(p_back, "post-lift tip retreat")
        ctx.log(f"tip was {tip_clear*1000:.1f} mm into/near board -> retreat "
                f"{np.linalg.norm(p_back - p_hold)*1000:.0f} mm along bit before reorient")
        robot.move_to_pose(p_back, R_hold, T=2.0)
        tip_clear = gates.tip_clearance(measured_tip(ctx), ctx.st["hole"], n)
        ctx.log(f"tip clearance after retreat: {tip_clear*1000:.1f} mm")
    ctx.st["holding"] = True
    ctx.phase_end("4/8 grasp", lift_mm=lifted * 1000.0, tip_clear_mm=tip_clear * 1000.0)
