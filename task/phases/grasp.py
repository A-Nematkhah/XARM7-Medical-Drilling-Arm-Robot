"""Phases 1–4: perceive the drill, grasp, lift, retreat the bit from the board."""
import os

import cv2
import numpy as np

from safety import GateError
from task.board_obs import (
    BODY_SAMPLES, BoardObstacle, poses_clearance, poses_hit,
)
from task.constants import (
    DRILL_SCAN_HI, DRILL_SCAN_LO, GRIP_BELOW_HOUSING, HAND_GRASP,
    SAFE_TIP, SLIDE, STANDOFF, TABLE_CLEAR_Z, TRANSIT_TCP_Z, UP,
)
from task.motion_ctrl import (
    hand_min_z as _hand_min_z,
    move_cart,
    reorient_to_pose as _reorient_to_hover,
    retreat_high as _retreat_high,
    safe_goto_joints,
    safe_goto_pose,
    table_margin as _table_margin,
)
from task.grasp_ctrl import CLOSE_SPEED, CLOSE_TIMEOUT, LOADED_IDX, close_until_load
from task.scene_est import update_scene_from_ctx
from task.state_machine import TaskState, TaskStateMachine
from task.tip import bind_tip_in_hand, log_tip_vs_gt, measured_tip


def perceive_and_grasp(ctx):
    """Detect from above, look from the side, go beside, rotate, slide in, grasp."""
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
    ws = _coarse_drill_xy(ctx)
    c_xy, ax = _scan_for_drill(ctx, ws)
    ctx.log(f"coarse: drill near ({c_xy[0]:.3f}, {c_xy[1]:.3f}), "
            f"axis yaw {np.degrees(np.arctan2(ax[1], ax[0])):+.1f} deg")
    ctx.metrics["coarse_xy_err_mm"] = round(float(np.linalg.norm(c_xy - ctx.gt_grip) * 1000.0), 2)
    ctx.phase_end("1/8 look: top-down scan", coarse_xy_err_mm=ctx.metrics["coarse_xy_err_mm"])

    ctx.phase("2/8 look: side view")
    side0 = np.cross(UP, [ax[0], ax[1], 0.0])
    sn = float(np.linalg.norm(side0))
    if sn < 1e-9:
        side0 = np.array([0.0, 1.0, 0.0])
    else:
        side0 = side0 / sn
    look = np.array([c_xy[0], c_xy[1], 0.11])
    targets = [look + [0, 0, dz] for dz in (-0.03, 0.0, 0.04, 0.09)] + [
        look + 0.12 * np.array([ax[0], ax[1], 0.8])]
    q = None
    ax3 = np.array([ax[0], ax[1], 0.0])
    view_sides = _side_signs_robot_left(ctx, look, side0, +0.34)
    for side_sign in view_sides:
        side = side_sign * side0
        for dist, hgt in [(0.34, 0.25), (0.30, 0.22), (0.38, 0.28)]:
            for up in (UP, -UP, ax3, -ax3):
                p, R = views.site_target_for_view(
                    look + dist * side + [0, 0, hgt], look, up_hint=up)
                try:
                    gates.check_workspace(p, "side-view")
                    gates.check_hand_vs_table(p, R, name="side-view")
                except GateError:
                    continue
                for cand in robot.ik_solutions(p, R):
                    if views.line_of_sight_clear(ctx.model, cand, targets, robot.qadr):
                        q = cand
                        break
                if q is not None:
                    break
            if q is not None:
                break
        if q is not None:
            break
    if q is None:
        raise RuntimeError("no reachable, unoccluded side view")
    safe_goto_joints(robot, q, gates=gates, T=ctx.motion["scan"].joint_T)
    robot.wait(0.4)
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

    a, bz, Rg, pg = _grasp_pose_from_est(est)
    ctx.tip_in_hand = bind_tip_in_hand(pg, Rg, est.bit_tip)
    ctx.a, ctx.Rg, ctx.pg = a, Rg, pg

    ctx.phase("3/8 approach (sideways, thumb pre-opposed)")
    ctx.hand.set_angles(ctx.hand.PRE_GRASP[:5] + [0], speed=800)
    a, bz, Rg, pg, slide, margin, p_slide = _approach_sideways(ctx, a, bz, Rg, pg)
    ctx.a, ctx.Rg, ctx.pg, ctx.p_slide = a, Rg, pg, p_slide
    robot.move_to_pose(pg, Rg, T=0.6)
    p_now, R_now = robot.site_pose()
    ctx.tip_in_hand = bind_tip_in_hand(p_now, R_now, est.bit_tip)
    ctx.phase_end("3/8 approach (sideways, thumb pre-opposed)",
                  slide_cm=slide * 100.0, joint_margin=margin,
                  track_err_mm=float(robot.max_track_err) * 1000.0)

    ctx.phase("4/8 grasp")
    sm = getattr(ctx, "state_machine", None) or TaskStateMachine()
    ctx.state_machine = sm
    sm.enter(TaskState.CLOSE_HAND, reason="at grip")
    # Free drill may have slid during approach. Small XY shifts: cart hop.
    # Large knocks: refresh axis from the live body and re-approach (wrap on
    # stale pg loads on air — seed 1 was 56 mm drift / 1 loaded finger).
    a, bz, Rg, pg, est = _refresh_grip_before_close(ctx, a, bz, Rg, pg, est)
    ctx.a, ctx.Rg, ctx.pg = a, Rg, pg
    close = _close_with_retry(ctx, pg, Rg, lateral=bz)
    p_now, R_now = robot.site_pose()
    t = bind_tip_in_hand(p_now, R_now, est.bit_tip)
    held = float(ctx.calib.tool.held_bit_reach)
    if held > 0:
        # Along-palm length after seating; keep lateral (y/z) from perception.
        t = np.asarray(t, float).copy()
        t[0] = np.sign(t[0] if abs(t[0]) > 1e-9 else 1.0) * held
    ctx.tip_in_hand = t
    ctx.log(f"tip_in_hand after grasp [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")
    log_tip_vs_gt(ctx, "post-close")
    f = close.forces
    frozen = [int(i) for i, flag in enumerate(close.frozen) if flag]
    ctx.log(f"finger torques [little ring middle index thumb_bend thumb_rot] = {np.round(f, 2)} N*m")
    ctx.log(f"grasp close: {close.n_loaded} loaded fingers after {close.elapsed:.2f}s "
            f"(frozen {frozen}, cmd {np.round(close.cmd).astype(int).tolist()})")
    # Deepen past the trip point so lift settle cannot unload wrap fingers.
    grasp_cmd = np.asarray(close.cmd, float).copy()
    target = np.asarray(HAND_GRASP, float).reshape(6)
    for i in LOADED_IDX:
        if i >= grasp_cmd.size:
            continue
        # Drive further toward the wrap target by ~80 units.
        if target[i] < grasp_cmd[i]:
            grasp_cmd[i] = max(float(target[i]), grasp_cmd[i] - 80.0)
        elif target[i] > grasp_cmd[i]:
            grasp_cmd[i] = min(float(target[i]), grasp_cmd[i] + 80.0)
    ctx.hand.set_angles(grasp_cmd, speed=CLOSE_SPEED)
    robot.wait(0.25)
    ctx.grasp_cmd = grasp_cmd
    sm.enter(TaskState.VERIFY_GRASP, reason="finger_load")
    try:
        gates.check_grasp(ctx.hand.get_forces())
    except GateError as ex:
        sm.abort(str(ex))
        ctx.metrics["grasp_verified"] = 0
        raise
    ctx.metrics["grasp_verified"] = 1
    if getattr(ctx, "scene", None) is not None:
        ctx.scene.grasp_verified = True
    sm.enter(TaskState.LIFT_AND_VERIFY, reason="verified")
    p_now, R_now = robot.site_pose()
    z0 = float(p_now[2])
    p_lift = p_now.copy()
    p_lift[2] = z0 + 0.15
    # Bake bit-away retreat into the lift so we do not stop at hover then yank back.
    n = ctx.st["board_n"]
    hole = ctx.st["hole"]
    tip = measured_tip(ctx)
    tip_clear = gates.tip_clearance(tip, hole, n)
    if _has_live_board_prior(ctx) and tip_clear < SAFE_TIP:
        bit = R_now[:, 0]
        pull = -bit if float(bit @ n) > 0.25 else -n
        dn_pull = float(pull @ n)
        if abs(dn_pull) < 0.2:
            pull, dn_pull = -n, -1.0
        s = (tip_clear - SAFE_TIP) / dn_pull
        p_back = p_now + s * pull
        p_back[2] = max(float(p_back[2]), float(p_lift[2]))
        g0 = gates.grasp_face_clearance(p_back, hole, n)
        if g0 < gates.REORIENT_GRASP_STANDOFF:
            p_back = p_back - (gates.REORIENT_GRASP_STANDOFF - g0) * n
            p_back[2] = max(float(p_back[2]), float(p_lift[2]))
        gates.check_workspace(p_back, "lift")
        ctx.log(f"tip was {tip_clear*1000:.1f} mm into/near board -> lift+retreat "
                f"{np.linalg.norm(p_back - p_now)*1000:.0f} mm")
        p_lift = p_back
    gates.check_workspace(p_lift, "lift")
    # Hold the deepened squeeze explicitly during the lift.
    ctx.hand.set_angles(grasp_cmd, speed=CLOSE_SPEED)
    lift_dist = float(np.linalg.norm(p_lift - p_now))
    robot.move_to_pose(p_lift, R_now, **ctx.motion["lift"].cart_kwargs(lift_dist))
    robot.wait(0.12)
    lifted = float(robot.site_pose()[0][2]) - z0
    ctx.log(f"lifted TCP by {lifted*1000:.0f} mm (commanded 150)")
    try:
        gates.check_lift(lifted)
    except GateError as ex:
        sm.abort(str(ex))
        ctx.metrics["lift_verified"] = 0
        raise
    f_lift = np.abs(np.asarray(ctx.hand.get_forces(), float))
    if int(np.sum(f_lift[list(LOADED_IDX)] >= gates.FINGER_LOAD)) < gates.MIN_LOADED_FINGERS:
        ctx.log("grasp unload after lift; re-squeeze")
        ctx.hand.set_angles(grasp_cmd, speed=CLOSE_SPEED)
        robot.wait(0.4)
        close2 = close_until_load(
            ctx.hand, robot, target=HAND_GRASP, load=gates.FINGER_LOAD,
            min_loaded=gates.MIN_LOADED_FINGERS, loaded_idx=LOADED_IDX,
            speed=CLOSE_SPEED, timeout=2.0)
        ctx.grasp_cmd = np.asarray(close2.cmd, float).copy()
        f_lift = close2.forces
    try:
        gates.check_grasp(f_lift)
    except GateError as ex:
        sm.abort(str(ex))
        ctx.metrics["lift_verified"] = 0
        raise
    # Sim: confirm the free drill rode up with the TCP (forces alone can trip on air).
    if data is not None:
        held_xy = np.asarray(data.body("drill").xpos[:2], float)
        tcp_xy = np.asarray(robot.site_pose()[0][:2], float)
        hold_mm = float(np.linalg.norm(held_xy - tcp_xy) * 1000.0)
        if hold_mm > 80.0:
            sm.abort("drill not held after lift")
            ctx.metrics["lift_verified"] = 0
            raise GateError(
                f"drill not held after lift ({hold_mm:.0f} mm from TCP); refuse")
    ctx.metrics["lift_verified"] = 1
    update_scene_from_ctx(ctx)

    p_hold, R_hold = robot.site_pose()
    bit = R_hold[:, 0]
    n = ctx.st["board_n"]
    hole = ctx.st["hole"]
    tip = measured_tip(ctx)
    tip_clear = gates.tip_clearance(tip, hole, n)
    if _has_live_board_prior(ctx) and tip_clear < SAFE_TIP:
        pull = -bit if float(bit @ n) > 0.25 else -n
        dn_pull = float(pull @ n)
        if abs(dn_pull) < 0.2:
            pull, dn_pull = -n, -1.0
        s = (tip_clear - SAFE_TIP) / dn_pull
        p_back = p_hold + s * pull
        p_back[2] = max(float(p_back[2]), float(p_hold[2]))
        g0 = gates.grasp_face_clearance(p_back, hole, n)
        if g0 < gates.REORIENT_GRASP_STANDOFF:
            p_back = p_back - (gates.REORIENT_GRASP_STANDOFF - g0) * n
            p_back[2] = max(float(p_back[2]), float(p_hold[2]))
        gates.check_workspace(p_back, "post-lift tip retreat")
        ctx.log(f"tip still {tip_clear*1000:.1f} mm into/near board -> extra retreat "
                f"{np.linalg.norm(p_back - p_hold)*1000:.0f} mm")
        retreat_dist = float(np.linalg.norm(p_back - p_hold))
        robot.move_to_pose(p_back, R_hold, **ctx.motion["relocate"].cart_kwargs(retreat_dist))
        tip_clear = gates.tip_clearance(measured_tip(ctx), hole, n)
        ctx.log(f"tip clearance after retreat: {tip_clear*1000:.1f} mm")
    log_tip_vs_gt(ctx, "post-lift")
    ctx.st["holding"] = True
    ctx.phase_end("4/8 grasp", lift_mm=lifted * 1000.0, tip_clear_mm=tip_clear * 1000.0)


def _drill_free_state(ctx):
    """Snapshot free-joint qpos/qvel for the drill (sim retry); None on hardware."""
    model = getattr(ctx, "model", None)
    data = getattr(ctx, "data", None)
    if model is None or data is None:
        return None
    try:
        import mujoco
        jid = int(model.joint("drill_free").id)
        adr = int(model.jnt_qposadr[jid])
        vadr = int(model.jnt_dofadr[jid])
        return (data.qpos[adr:adr + 7].copy(), data.qvel[vadr:vadr + 6].copy(), adr, vadr)
    except Exception:
        return None


def _restore_drill_free(ctx, snap):
    if snap is None:
        return
    data = getattr(ctx, "data", None)
    model = getattr(ctx, "model", None)
    if data is None or model is None:
        return
    q, v, adr, vadr = snap
    data.qpos[adr:adr + 7] = q
    data.qvel[vadr:vadr + 6] = v
    try:
        import mujoco
        mujoco.mj_forward(model, data)
    except Exception:
        pass


def _drill_knock_mm(ctx):
    """XY drift of the free drill vs pre-approach ground truth (mm); 0 if no sim data."""
    data = getattr(ctx, "data", None)
    gt = getattr(ctx, "gt_grip", None)
    if data is None or gt is None:
        return 0.0
    try:
        return float(np.linalg.norm(data.body("drill").xpos[:2] - np.asarray(gt, float)[:2]) * 1000.0)
    except Exception:
        return 0.0


def _assert_drill_stable(ctx, limit_mm=25.0, where="approach"):
    dist_mm = _drill_knock_mm(ctx)
    if dist_mm <= float(limit_mm):
        return dist_mm
    raise GateError(f"drill knocked during {where} ({dist_mm:.0f} mm); refuse grasp")


def _settle_joints(robot, q, T=1.2, tol=0.06, retries=2, gates=None):
    """Drive to `q` and re-issue if measured joints lag the command."""
    q = np.asarray(q, float).reshape(7)
    safe_goto_joints(robot, q, gates=gates, T=T)
    for _ in range(int(retries)):
        if float(np.linalg.norm(robot.get_q() - q)) < float(tol):
            break
        safe_goto_joints(robot, q, gates=gates, T=max(0.5, 0.5 * float(T)))
    if hasattr(robot, "q_cmd"):
        robot.q_cmd = robot.get_q()


def _joint_to_pose(robot, p_tgt, Rg, q_seed, T, tag, limit=0.080, prefer_seed_q=True):
    """Move to pose. Planned q first when reachable; else re-IK from live q.

    Seed-3 taught us that continuous-path `chain[k]` can be an unreachable
    branch under contact even when offline FK at that q is perfect — re-solving
    from the live configuration finds a reachable solution.
    """
    p_tgt = np.asarray(p_tgt, float)
    if prefer_seed_q and q_seed is not None:
        _settle_joints(robot, q_seed, T=T, retries=3)
        err = float(np.linalg.norm(robot.site_pose()[0] - p_tgt))
        if err <= limit:
            return err
    try:
        safe_goto_pose(robot, p_tgt, Rg, q_goal=q_seed, T_joint=T)
    except RuntimeError as ex:
        raise GateError(
            f"approach joint residual at {tag}; IK reacquire failed ({ex})"
        ) from ex
    err = float(np.linalg.norm(robot.site_pose()[0] - p_tgt))
    if err > limit:
        raise GateError(
            f"approach joint residual {err*1000:.0f} mm at {tag}; "
            "IK chain missed approach")
    return err


def _execute_grasp_approach(ctx, robot, pg, a, Rg, wps, chain, p_slide, q_start):
    """Joint via+entry, Cartesian slide; raised-grip fallback if cart stalls.

    Seed 3/6: offline chain low qs are often dynamically unreachable, but a
    live-seeded Cartesian from entry (or raised-grip re-IK) still works when it
    does not shove the free drill.
    """
    del q_start  # caller restores on failure
    gates = ctx.gates
    gprof = ctx.motion["grasp"]
    pg = np.asarray(pg, float)
    p_slide = np.asarray(p_slide, float)
    p_stand = pg - STANDOFF * np.asarray(a, float)
    if not chain:
        raise GateError("empty grasp IK chain")
    p_via = np.asarray(wps[0], float)
    p_entry = np.asarray(wps[1] if len(wps) > 1 else p_slide, float)
    T0 = max(float(gprof.joint_T_first), 2.0)
    T = max(float(gprof.joint_T_step), 1.5)

    via_err = _joint_to_pose(robot, p_via, Rg, chain[0], T0, "high via")
    _assert_drill_stable(ctx, limit_mm=35.0, where="approach")
    entry_err = 0.0
    if len(chain) >= 2:
        entry_err = _joint_to_pose(
            robot, p_entry, Rg, chain[1], T, "side entry")
        _assert_drill_stable(ctx, limit_mm=35.0, where="approach")

    p_above = p_slide.copy()
    p_above[2] = max(float(p_slide[2]) + 0.10, float(pg[2]) + 0.12, float(p_entry[2]))
    peak = max(float(via_err), float(entry_err))
    cart_ok = True
    for tgt, vmax, hop, tag in (
        (p_above, 0.04, 0.03, "above-slide"),
        (p_slide, 0.028, 0.02, "slide"),
        (p_stand, 0.022, 0.015, "standoff"),
        (pg, 0.015, 0.012, "grip"),
    ):
        peak = max(peak, _slow_cart_to(robot, tgt, Rg, vmax=vmax, hop=hop))
        _assert_drill_stable(ctx, limit_mm=35.0, where="approach")
        err = float(np.linalg.norm(robot.site_pose()[0] - np.asarray(tgt, float)))
        if err > 0.045:
            ctx.log(f"cart residual {err*1000:.0f} mm at {tag}; joint reacquire")
            try:
                safe_goto_pose(
                    robot, tgt, Rg, T_joint=max(float(gprof.joint_T_step), 1.2),
                    gates=gates)
            except RuntimeError:
                cart_ok = False
                break
            err = float(np.linalg.norm(robot.site_pose()[0] - np.asarray(tgt, float)))
            peak = max(peak, err)
        if err > 0.080:
            cart_ok = False
            break

    residual = float(np.linalg.norm(robot.site_pose()[0] - pg))
    # Seed-3 recovery: Cartesian from entry often stalls ~150 mm; re-IK a raised
    # grip from the live q, then drop (only if we have not already shoved hard).
    if (not cart_ok or residual > 0.025) and _drill_knock_mm(ctx) < 15.0:
        ctx.log("cart approach stalled; raised-grip fallback")
        p_hi = p_slide.copy()
        p_hi[2] = max(float(p_slide[2]) + 0.14, float(pg[2]) + 0.16)
        try:
            peak = max(peak, _joint_to_pose(
                robot, p_hi, Rg, None, T, "above-slide", prefer_seed_q=False,
                limit=0.100))
            for dz in (0.10, 0.08, 0.06):
                p_raised = pg.copy()
                p_raised[2] = float(pg[2]) + float(dz)
                try:
                    peak = max(peak, _joint_to_pose(
                        robot, p_raised, Rg, None, T, "raised-grip",
                        prefer_seed_q=False, limit=0.100))
                    break
                except GateError:
                    continue
            peak = max(peak, _slow_cart_to(robot, pg, Rg, vmax=0.012, hop=0.008))
            residual = float(np.linalg.norm(robot.site_pose()[0] - pg))
            if residual > 0.025:
                safe_goto_pose(
                    robot, pg, Rg, T_joint=max(float(gprof.joint_T_step), 1.2),
                    gates=gates)
                residual = float(np.linalg.norm(robot.site_pose()[0] - pg))
        except (GateError, RuntimeError) as ex:
            raise GateError(
                f"approach joint residual at raised-grip; {ex}") from ex

    if residual > 0.080:
        raise GateError(
            f"approach joint residual {residual*1000:.0f} mm at grip; "
            "IK chain missed approach")

    q_meas = robot.get_q()
    robot.max_track_err = max(residual, peak)
    ctx.log(f"slide residual {residual*1000:.1f} mm, cart peak {peak*1000:.1f} mm; "
            f"q={np.round(q_meas, 2)}")
    dist_mm = _drill_knock_mm(ctx)
    if dist_mm > 0.0:
        ctx.log(f"drill disturbance during approach: {dist_mm:.1f} mm")
    _assert_drill_stable(ctx, limit_mm=35.0, where="approach")
    if residual > 0.025 and peak > 0.045 and dist_mm > 8.0:
        raise GateError(
            f"approach cart peak {peak*1000:.0f} mm with "
            f"{dist_mm:.0f} mm drill drift; path lost tracking")
    if dist_mm > 18.0:
        raise GateError(
            f"drill knocked during approach ({dist_mm:.0f} mm); refuse grasp")
    gates.check_tracking(residual, "slide")
    gates.check_joint_margin(robot.limit_margin(q_meas))
    return residual


def _slow_cart_to(robot, p_goal, R_goal, vmax=0.03, hop=0.03, settle=0.12):
    """Short Cartesian hops so a large joint-path residual does not one-shot the gate."""
    p_goal = np.asarray(p_goal, float)
    p0 = np.asarray(robot.site_pose()[0], float)
    dist = float(np.linalg.norm(p_goal - p0))
    if dist < 1e-4:
        return 0.0
    nseg = max(1, int(np.ceil(dist / max(float(hop), 1e-3))))
    peak = 0.0
    for i in range(1, nseg + 1):
        pi = p_goal if i == nseg else p0 + (i / nseg) * (p_goal - p0)
        d = float(np.linalg.norm(pi - robot.site_pose()[0]))
        T = max(0.6, d / max(float(vmax), 1e-6))
        robot.move_to_pose(pi, R_goal, T=T, vmax=vmax, settle=settle)
        peak = max(peak, float(robot.max_track_err))
    return peak


# Classic long side-slides first (working seeds). Short slides cover +Y
# workspace clips and leftover IK misses (seeds 1 / 3).
_APPROACH_CLASSIC = (
    (SLIDE, 0.06, 0.10),
    (0.12, 0.04, 0.08),
    (0.12, 0.06, 0.05),
    (0.12, 0.01, 0.03),
    (0.12, 0.0, 0.02),
    (0.10, 0.03, 0.07),
)
_APPROACH_SHORT = (
    (0.08, 0.03, 0.08),
    (0.08, 0.02, 0.05),
    (0.06, 0.02, 0.06),
)


def _approach_option_list(prefer_short=False):
    if prefer_short:
        return list(_APPROACH_SHORT) + list(_APPROACH_CLASSIC)
    return list(_APPROACH_CLASSIC) + list(_APPROACH_SHORT)


def _grasp_pose_from_est(est):
    a = np.asarray(est.axis, float)
    nrm = float(np.linalg.norm(a))
    a = a / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0])
    bz = np.cross(a, UP)
    bn = float(np.linalg.norm(bz))
    if bn < 1e-9:
        bz = np.array([0.0, 1.0, 0.0])
    else:
        bz = bz / bn
    Rg = np.column_stack([a, UP, bz])
    pg = np.array([*est.grip_xy, est.housing_underside_z - GRIP_BELOW_HOUSING])
    return a, bz, Rg, pg


def _clipped_side_wps(ctx, pg, a, bz_s, slide, out, up):
    """Same polyline as the live approach, with only the high entry clipped."""
    pg = np.asarray(pg, float)
    a = np.asarray(a, float)
    bz_s = np.asarray(bz_s, float)
    p_slide = pg - float(slide) * bz_s - STANDOFF * a
    p_entry = p_slide - float(out) * bz_s + np.array([0.0, 0.0, float(up)])
    p_entry = _clip_to_workspace(ctx, p_entry)
    wps = ([p_entry, p_slide]
           + [pg - STANDOFF * a - s * float(slide) * bz_s for s in (0.66, 0.33, 0.0)]
           + [pg])
    return wps, p_slide


def _shoulder_folded(q):
    """True when J2 is cranked down; Cartesian hops from here miss by ~25 mm."""
    return float(np.asarray(q, float).reshape(-1)[1]) < -1.0


def _iter_side_geoms(ctx, pg, a, bz, side_sign, options):
    """Clip / via / drop polylines for one approach side."""
    bz_s = float(side_sign) * np.asarray(bz, float)
    seen_drop = set()
    for kind in ("clip", "via", "drop"):
        for slide, out, up in options:
            if kind == "clip":
                wps, p_slide = _clipped_side_wps(ctx, pg, a, bz_s, slide, out, up)
            elif kind == "via":
                wps, p_slide = _grasp_waypoints(ctx, pg, a, bz_s, slide, out, up)
            else:
                key = (round(float(slide), 3), round(float(out), 3))
                if key in seen_drop:
                    continue
                seen_drop.add(key)
                wps, p_slide = _side_slide_waypoints(pg, a, bz_s, slide, out)
                wps = [np.asarray(p, float).copy() for p in wps]
                wps[0] = _clip_to_workspace(ctx, wps[0])
            yield float(side_sign), float(slide), float(out), float(up), wps, p_slide, kind


def _approach_sideways(ctx, a, bz, Rg, pg, prefer_short=False):
    """Side-slide to `pg` from the robot's left (+Y) only."""
    robot, gates = ctx.robot, ctx.gates
    gates.check_workspace(pg, "grasp")
    q_start = robot.get_q()
    drill_snap = _drill_free_state(ctx)
    last_err = None
    tried_motion = set()
    gprof = ctx.motion["grasp"]

    def _restore_start():
        try:
            safe_goto_joints(
                robot, q_start, gates=gates, T=ctx.motion["scan"].joint_T)
        except (GateError, RuntimeError) as ex:
            ctx.log(f"approach restore: {ex}")
        _restore_drill_free(ctx, drill_snap)
    frames = [(np.asarray(a, float), np.asarray(bz, float), np.asarray(Rg, float), "primary")]

    def _run_chain(side_sign, slide, out, up, wps, p_slide, a_use, Rg_use, how,
                   allow_folded=True, careful=False):
        nonlocal last_err
        key = (round(float(side_sign), 0), round(float(slide), 3),
               round(float(out), 3), round(float(up), 3), how)
        if key in tried_motion:
            return None
        try:
            _gate_workspace(ctx, wps)
            _gate_transit_clears_table([wps[0]], Rg_use, gates=gates)
            cands = list(_ik_candidates(robot, wps, Rg_use))
        except (RuntimeError, GateError) as ex:
            ctx.log(f"approach option side={side_sign:+.0f} {slide, out, up} "
                    f"[{how}] rejected: {ex}")
            last_err = ex
            return None
        unfolded = [t for t in cands if not _shoulder_folded(t[1][-1])]
        if not unfolded and not allow_folded:
            ctx.log(f"approach option side={side_sign:+.0f} {slide, out, up} "
                    f"[{how}] rejected: only folded-shoulder IK")
            last_err = GateError("only folded-shoulder IK")
            return None
        use = (unfolded if unfolded else list(cands))[:2]
        side_y = float(np.asarray(p_slide, float)[1] - np.asarray(pg, float)[1])
        for _cost, chain, margin in use:
            try:
                gates.check_joint_margin(margin)
            except GateError as ex:
                last_err = ex
                continue
            ctx.log(f"approach: side {side_sign:+.0f} "
                    f"({'robot-left' if side_y >= 0.0 else 'robot-right'}), "
                    f"slide {slide*100:.0f} cm, {how}, "
                    f"joint-limit margin {margin:.2f} rad")
            tried_motion.add(key)
            try:
                if careful:
                    _execute_grasp_approach(
                        ctx, robot, pg, a_use, Rg_use, wps, chain, p_slide,
                        q_start)
                    q_meas = robot.get_q()
                    residual = float(np.linalg.norm(robot.site_pose()[0] - pg))
                    ctx.log(f"slide residual {residual*1000:.1f} mm; "
                            f"q={np.round(q_meas, 2)}")
                else:
                    robot.move_joint_path(
                        chain[:-1],
                        T_first=max(float(gprof.joint_T_first), 3.0),
                        T_step=max(float(gprof.joint_T_step), 0.8))
                    gates.check_joint_margin(robot.limit_margin(robot.get_q()))
                    e = robot.move_to_pose(pg - STANDOFF * a_use, Rg_use, T=1.5)
                    q_meas = robot.get_q()
                    ctx.log(f"slide max tracking err {robot.max_track_err*1000:.1f} mm; "
                            f"residual {np.round(e[:3]*1000, 1)} mm; "
                            f"q={np.round(q_meas, 2)}")
                    gates.check_tracking(robot.max_track_err, "slide")
                    gates.check_joint_margin(robot.limit_margin(q_meas))
                knock = _drill_knock_mm(ctx)
                if knock > 18.0:
                    ctx.log(f"approach option side={side_sign:+.0f} {slide, out, up} "
                            f"[{how}] knocked the drill {knock:.0f} mm; "
                            "restore and try next")
                    last_err = GateError(
                        f"drill knocked during approach ({knock:.0f} mm)")
                    _restore_start()
                    return None
                return slide, margin, p_slide
            except GateError as ex:
                last_err = ex
                ctx.log(f"approach option side={side_sign:+.0f} {slide, out, up} "
                        f"[{how}] failed in motion: {ex}")
                _restore_start()
                return None
        if not use:
            last_err = last_err or GateError("no IK chain with joint-limit margin")
        return None

    def _try_frame(fa, fbz, fRg, lab, emergency, allow_folded=True):
        nonlocal last_err
        side_order = _grasp_side_left_only(ctx, pg, fbz, -SLIDE)
        for side_sign in side_order:
            opts = _approach_option_list(prefer_short=prefer_short)
            n_knock = 0
            for side_s, slide, out, up, wps, p_slide, kind in _iter_side_geoms(
                    ctx, pg, fa, fbz, side_sign, opts):
                how = lab if kind == "clip" and lab != "primary" else kind
                got = _run_chain(side_s, slide, out, up, wps, p_slide, fa, fRg, how,
                                 allow_folded=allow_folded, careful=prefer_short)
                if got is not None:
                    return (fa, fbz, fRg, pg) + got
                if last_err is not None and "knocked" in str(last_err).lower():
                    n_knock += 1
                    if n_knock >= 2:
                        ctx.log(f"approach: two knocks on robot-left; "
                                "skip remaining geometries on this side")
                        break
        if not emergency:
            return None
        emerg = _list_emergency_raised_paths(
            ctx, robot, pg, fa, fbz, fRg, side_order, None, limit=2)
        for slide, out, up, side_sign, wps, chain, margin, p_slide, _clr in emerg:
            ctx.log(f"approach: emergency side {side_sign:+.0f} "
                    f"slide {slide*100:.0f} cm, joint-limit margin {margin:.2f} rad")
            try:
                _execute_grasp_approach(
                    ctx, robot, pg, fa, fRg, wps, chain, p_slide, q_start)
                return fa, fbz, fRg, pg, float(slide), float(margin), p_slide
            except (GateError, RuntimeError) as ex:
                last_err = ex
                ctx.log(f"approach option side={side_sign:+.0f} {slide, out, up} "
                        f"[emergency] failed in motion: {ex}")
                _restore_start()
        return None

    for fa, fbz, fRg, lab in frames:
        got = _try_frame(fa, fbz, fRg, lab, emergency=False, allow_folded=False)
        if got is not None:
            return got
    got = _try_frame(frames[0][0], frames[0][1], frames[0][2], frames[0][3],
                     emergency=False, allow_folded=True)
    if got is not None:
        return got
    got = _try_frame(frames[0][0], frames[0][1], frames[0][2], frames[0][3],
                     emergency=True, allow_folded=True)
    if got is not None:
        return got

    raise last_err if last_err is not None else GateError(
        "drill is outside the reachable grasping workspace")


def _grip_from_live_drill(ctx, est, pg, a):
    """Rebuild grasp pose from the sim drill body after a knock. Hardware: no-op."""
    data = getattr(ctx, "data", None)
    if data is None:
        return _grasp_pose_from_est(est) + (est,)
    live_xy = np.asarray(data.body("drill").xpos[:2], float)
    Rdr = np.asarray(data.body("drill").xmat, float).reshape(3, 3)
    ax = np.array([Rdr[0, 0], Rdr[1, 0], 0.0], float)
    nrm = float(np.linalg.norm(ax))
    if nrm < 1e-9:
        ax = np.array([a[0], a[1], 0.0], float)
        nrm = float(np.linalg.norm(ax))
    ax = ax / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0])
    if float(ax[:2] @ np.asarray(a, float)[:2]) < 0.0:
        ax = -ax
    bz = np.cross(ax, UP)
    bn = float(np.linalg.norm(bz))
    bz = bz / bn if bn > 1e-9 else np.array([0.0, 1.0, 0.0])
    Rg = np.column_stack([ax, UP, bz])
    pg2 = np.array([live_xy[0], live_xy[1], float(pg[2])], float)
    tip = np.asarray(est.bit_tip, float).copy()
    site = getattr(ctx, "tip_site", None)
    if site is not None and int(site) >= 0:
        try:
            tip = np.asarray(data.site_xpos[int(site)], float).copy()
        except Exception:
            pass
    est.grip_xy = live_xy.copy()
    est.axis = ax
    est.bit_tip = tip
    return ax, bz, Rg, pg2, est


def _refresh_grip_before_close(ctx, a, bz, Rg, pg, est):
    """After approach, follow a knocked drill instead of wrapping on stale pg."""
    robot, gates = ctx.robot, ctx.gates
    dist_mm = _drill_knock_mm(ctx)
    data = getattr(ctx, "data", None)
    if dist_mm <= 5.0 or data is None:
        return a, bz, Rg, pg, est
    live_xy = np.asarray(data.body("drill").xpos[:2], float)
    if float(np.linalg.norm(live_xy - np.asarray(pg[:2], float))) <= 0.004:
        return a, bz, Rg, pg, est

    a2, bz2, Rg2, pg2, est2 = _grip_from_live_drill(ctx, est, pg, a)
    if dist_mm <= 15.0:
        ctx.log(f"reacquire live grip XY after {dist_mm:.0f} mm drift "
                f"-> ({pg2[0]:.3f}, {pg2[1]:.3f})")
        gates.check_workspace(pg2, "grasp-reacquire")
        peak = _slow_cart_to(robot, pg2 - STANDOFF * a2, Rg2, vmax=0.025, hop=0.02)
        peak = max(peak, _slow_cart_to(robot, pg2, Rg2, vmax=0.015, hop=0.012))
        residual = float(np.linalg.norm(robot.site_pose()[0] - pg2))
        robot.max_track_err = max(float(robot.max_track_err), residual, peak)
        gates.check_tracking(residual, "grasp-reacquire")
        return a2, bz2, Rg2, pg2, est2

    ctx.log(f"large drill drift {dist_mm:.0f} mm; re-approach from live pose "
            f"-> ({pg2[0]:.3f}, {pg2[1]:.3f})")
    ctx.grasp_reapproached = True
    ctx.hand.set_angles(ctx.hand.PRE_GRASP[:5] + [0], speed=800)
    p_now, R_now = robot.site_pose()
    p_up = p_now.copy()
    p_up[2] = max(float(p_now[2]) + 0.10, TRANSIT_TCP_Z)
    try:
        gates.check_workspace(p_up, "knock-lift")
        gates.check_hand_vs_table(p_up, R_now, name="knock-lift")
        _slow_cart_to(robot, p_up, R_now, vmax=0.04, hop=0.03)
    except GateError as ex:
        ctx.log(f"knock-lift skipped: {ex}")
    a2, bz2, Rg2, pg2, slide, margin, p_slide = _approach_sideways(
        ctx, a2, bz2, Rg2, pg2, prefer_short=True)
    del slide, margin
    ctx.p_slide = p_slide
    robot.move_to_pose(pg2, Rg2, T=0.6)
    return a2, bz2, Rg2, pg2, est2


def _close_with_retry(ctx, pg, Rg, lateral=None):
    """Close until loaded; nudge Z / along-axis / lateral if wrap fingers miss."""
    robot, gates = ctx.robot, ctx.gates
    close = close_until_load(
        ctx.hand, robot, target=HAND_GRASP, load=gates.FINGER_LOAD,
        min_loaded=gates.MIN_LOADED_FINGERS, loaded_idx=LOADED_IDX,
        speed=CLOSE_SPEED, timeout=CLOSE_TIMEOUT)
    if close.n_loaded >= gates.MIN_LOADED_FINGERS:
        return close
    # 0–1 loaded fingers means the hand closed on air. Millimetre nudges
    # around a stale grip cannot create wrap contact (seed 1: 1 finger / 335 mm).
    if close.n_loaded < 2 or getattr(ctx, "grasp_reapproached", False):
        ctx.log(f"grasp retry skipped: {close.n_loaded} loaded fingers "
                "(need a wrap, not air nudges)")
        return close
    ctx.log(f"grasp retry: {close.n_loaded} loaded fingers, reseat nudges")
    lat = None
    if lateral is not None:
        v = np.asarray(lateral, float).reshape(3)
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-9:
            lat = v / nrm
    a_col = np.asarray(Rg, float)[:, 0]
    nudges = [
        np.array([0.0, 0.0, 0.008]),
        np.array([0.0, 0.0, -0.004]),
        0.006 * a_col,
        -0.004 * a_col,
    ]
    if lat is not None:
        nudges.extend([0.006 * lat, -0.006 * lat])
    for dlt in nudges:
        p_nudge = np.asarray(pg, float) + dlt
        try:
            gates.check_workspace(p_nudge, "grasp-retry")
            robot.move_to_pose(
                p_nudge, Rg, **ctx.motion["grasp"].cart_kwargs(float(np.linalg.norm(dlt))))
        except GateError as ex:
            ctx.log(f"grasp retry skipped: {ex}")
            continue
        close = close_until_load(
            ctx.hand, robot, target=HAND_GRASP, load=gates.FINGER_LOAD,
            min_loaded=gates.MIN_LOADED_FINGERS, loaded_idx=LOADED_IDX,
            speed=CLOSE_SPEED, timeout=3.0)
        if close.n_loaded >= gates.MIN_LOADED_FINGERS:
            return close
    return close


def _scan_for_drill(ctx, ws):
    """Top-down D435i look; if LiDAR missed, spiral over the table spawn box."""
    robot, views, gates, cam, det = (
        ctx.robot, ctx.views, ctx.gates, ctx.cam, ctx.det)
    capture = getattr(cam, "settle_capture", None) or cam.capture
    looks = [np.asarray(ws[:2], float)]
    if getattr(getattr(ctx, "coarse_map", None), "drill_xy", None) is None:
        xs = np.linspace(DRILL_SCAN_LO[0], DRILL_SCAN_HI[0], 3)
        ys = np.linspace(DRILL_SCAN_LO[1], DRILL_SCAN_HI[1], 3)
        for y in ys:
            for x in xs:
                cand = np.array([float(x), float(y)])
                if float(np.linalg.norm(cand - looks[0])) > 0.04:
                    looks.append(cand)
    last_err = None
    for i, xy in enumerate(looks):
        look = np.array([xy[0], xy[1], 0.0])
        p, R = views.site_target_for_view(
            look + np.array([-0.05, 0.0, 0.60]), look, up_hint=(0, 1, 0))
        try:
            gates.check_workspace(p, "scan")
            gates.check_hand_vs_table(p, R, name="scan")
        except GateError:
            continue
        safe_goto_pose(
            robot, p, R, gates=gates, T_joint=ctx.motion["scan"].joint_T)
        robot.wait(0.35)
        rgb, dep = capture()
        try:
            c_xy, ax = det.detect_coarse(rgb, dep, cam.pose())
        except RuntimeError as e:
            last_err = e
            ctx.log(f"scan look {i} at ({xy[0]:.2f}, {xy[1]:.2f}): {e}")
            continue
        ctx.show_detection(rgb, 1.5)
        if i > 0:
            ctx.log(f"drill found on scan look {i} at ({xy[0]:.2f}, {xy[1]:.2f})")
        return c_xy, ax
    raise RuntimeError(str(last_err) if last_err else "drill not found")


def _coarse_drill_xy(ctx):
    """One 360° wrist LiDAR scan → look-at for the D435i (go above that XY)."""
    ws = np.array([0.40, 0.0, 0.0])
    lidar = getattr(ctx, "lidar", None)
    if lidar is None:
        return ws
    try:
        cmap = lidar.locate()
    except Exception as e:
        ctx.log(f"lidar coarse miss ({e}); wrist scans table grid")
        return ws
    ctx.coarse_map = cmap
    ctx.st["lidar_scan"] = getattr(lidar, "last_scan", None)
    ctx.st["lidar_map"] = cmap
    if cmap.drill_xy is not None:
        err = float(np.linalg.norm(cmap.drill_xy - ctx.gt_grip[:2])) * 1000.0
        ctx.metrics["lidar_drill_err_mm"] = round(err, 2)
        ctx.log(f"lidar drill ~ ({cmap.drill_xy[0]:.3f}, {cmap.drill_xy[1]:.3f}) "
                f"err {err:.0f} mm vs spawn")
        ws = np.array([cmap.drill_xy[0], cmap.drill_xy[1], 0.0])
    else:
        ctx.log("lidar found no drill cluster; wrist will spiral-search the table")
    if cmap.board_xy is not None:
        ctx.log(f"lidar board ~ ({cmap.board_xy[0]:.3f}, {cmap.board_xy[1]:.3f})")
        if ctx.hole_gt is not None:
            ctx.metrics["lidar_board_err_mm"] = round(
                float(np.linalg.norm(cmap.board_xy - ctx.hole_gt[:2])) * 1000.0, 2)
        # Seed a coarse face prior so grasp approach / tip retreat avoid the board
        # before Phase 5 refreshes from the wrist D435i.
        _seed_board_from_lidar(ctx, cmap)
    ctx.runlog.event("1/8 look: top-down scan", "lidar",
                     drill_xy=None if cmap.drill_xy is None else cmap.drill_xy.tolist(),
                     board_xy=None if cmap.board_xy is None else cmap.board_xy.tolist())
    return ws


def _seed_board_from_lidar(ctx, cmap):
    """Replace calib hole/normal with a coarse LiDAR face prior (not a pierce target)."""
    if cmap.board_xy is None:
        return
    pts = getattr(cmap, "board_pts", None)
    if pts is not None and len(np.asarray(pts).reshape(-1, 3)) >= 3:
        P = np.asarray(pts, float).reshape(-1, 3)
        face = P.mean(0)
    else:
        z = float(ctx.hole_hint[2]) if ctx.hole_hint is not None else 0.34
        face = np.array([float(cmap.board_xy[0]), float(cmap.board_xy[1]), z])
    n = cmap.board_n
    if n is None:
        n = face.copy(); n[2] = 0.0
        if np.linalg.norm(n) < 1e-6:
            n = np.asarray(ctx.board_n_hint, float).copy()
        else:
            n = n / np.linalg.norm(n)
    else:
        n = np.asarray(n, float)
        n = n / (np.linalg.norm(n) + 1e-12)
    ctx.st["board_n"] = n
    ctx.st["hole"] = face
    ctx.log(f"lidar board prior: n=({n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f}) "
            f"at ({face[0]:.3f}, {face[1]:.3f}, {face[2]:.3f})")


def _has_live_board_prior(ctx):
    """True when LiDAR (not the calib hole hint) has a face XY for post-grasp retreat."""
    cmap = getattr(ctx, "coarse_map", None)
    return cmap is not None and getattr(cmap, "board_xy", None) is not None


def _board_xy_prior(ctx):
    cmap = getattr(ctx, "coarse_map", None)
    if cmap is not None and cmap.board_xy is not None:
        return np.asarray(cmap.board_xy, float).reshape(2)
    hole = ctx.st.get("hole")
    if hole is not None:
        return np.asarray(hole, float).reshape(3)[:2]
    if ctx.hole_hint is not None:
        return np.asarray(ctx.hole_hint, float).reshape(3)[:2]
    return None


def _board_plane(ctx):
    """(hole, normal, half) for geometric hand-vs-board checks."""
    hole = ctx.st.get("hole")
    n = ctx.st.get("board_n")
    if hole is None or n is None:
        hole = ctx.hole_hint
        n = ctx.board_n_hint
    if hole is None or n is None:
        return None
    half = ctx.calib.board.half if ctx.calib is not None else None
    return np.asarray(hole, float), np.asarray(n, float), half


def _side_signs_robot_left(ctx, origin, lateral, offset):
    """`+1/-1` preferring the robot's left (+Y) standoff; opposite is last.

    xArm faces +X toward the board, so left-hand approach is the larger world-Y
    of `origin + offset*sign*lateral`. Side-view look may use both; grasp
    approach uses only the first sign via `_grasp_side_left_only`.
    """
    del ctx  # kept for call-site symmetry with the old board-away helper
    sides = [1.0, -1.0]
    o = np.asarray(origin, float).reshape(-1)[:2]
    lat = np.asarray(lateral, float).reshape(-1)[:2]
    ln = float(np.linalg.norm(lat))
    if ln < 1e-9:
        return sides
    lat = lat / ln

    def world_y(sign):
        p = o + float(offset) * float(sign) * lat
        return float(p[1])

    return sorted(sides, key=world_y, reverse=True)


def _grasp_side_left_only(ctx, origin, lateral, offset):
    """Robot-left standoff only. Grasp never slides in from the right."""
    return [_side_signs_robot_left(ctx, origin, lateral, offset)[0]]


def _side_signs_away_from_board(ctx, origin, lateral, offset):
    """Deprecated alias: grasp always prefers robot-left now."""
    return _side_signs_robot_left(ctx, origin, lateral, offset)


def _board_away_normal(ctx):
    plane = _board_plane(ctx)
    if plane is None:
        return None
    _, n, _ = plane
    n = np.asarray(n, float)
    n = n / (np.linalg.norm(n) + 1e-12)
    return n


def _rot_about(axis, deg):
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    th = np.deg2rad(float(deg))
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _grasp_frames(axis, rolls_deg=(0.0, 20.0, -20.0, 35.0, -35.0)):
    """Grasp orientations that track the top-down drill yaw.

    Flattens the bit axis to XY, tries both bit directions, then optional wrist
    rolls about that axis (palm Y stays near world-up). Live approach uses
    roll=0 from the measured yaw; rolls are a fallback for callers that need them.
    Returns list of (a, bz, Rg, label).
    """
    a0 = np.asarray(axis, float).reshape(3)
    a_xy = np.array([a0[0], a0[1], 0.0], float)
    if float(np.linalg.norm(a_xy)) < 1e-6:
        a_xy = np.array([1.0, 0.0, 0.0])
    else:
        a_xy = a_xy / np.linalg.norm(a_xy)
    frames = []
    for sign in (+1.0, -1.0):
        ax = sign * a_xy
        bz0 = np.cross(ax, UP)
        bn = float(np.linalg.norm(bz0))
        if bn < 1e-9:
            continue
        bz0 = bz0 / bn
        Rg0 = np.column_stack([ax, UP, bz0])
        for roll in rolls_deg:
            if abs(float(roll)) < 1e-9:
                frames.append((ax, bz0, Rg0.copy(), f"bit{sign:+.0f}_r0"))
                continue
            R = _rot_about(ax, roll) @ Rg0
            # Keep columns orthonormal; require palm-Y still mostly world-up
            # so pinky/LiDAR stay above the table during transit.
            y = R[:, 1]
            if abs(float(y @ UP)) < 0.75:
                continue
            z = R[:, 2]
            z = z / (np.linalg.norm(z) + 1e-12)
            y = np.cross(z, ax)
            y = y / (np.linalg.norm(y) + 1e-12)
            z = np.cross(ax, y)
            Rg = np.column_stack([ax, y, z])
            frames.append((ax, z, Rg, f"bit{sign:+.0f}_r{roll:+.0f}"))
    return frames


def _gate_transit_clears_table(wps, R, table_z=0.0, margin=TABLE_CLEAR_Z, gates=None):
    floor = float(table_z) + _table_margin(gates, margin)
    if gates is not None:
        for i, p in enumerate(wps):
            gates.check_hand_vs_table(p, R, name=f"grasp-wp{i}")
        return
    for i, p in enumerate(wps):
        zmin = _hand_min_z(p, R)
        if zmin < floor:
            raise GateError(
                f"grasp-wp{i} hand would hit the table "
                f"(sample z {zmin*1000:.0f} mm)")


def _side_slide_waypoints(pg, a, bz_s, slide, out, transit_z=TRANSIT_TCP_Z):
    """High hover → lateral at height → vertical drop → horizontal slide in.

    Returns (waypoints, p_slide). Transit Z stays high so pinky/LiDAR (~8 cm
    below TCP when grasp Y is world-up) clears the table; only the last three
    poses sit at grip height. The drop is purely vertical (same XY as p_slide).
    """
    pg = np.asarray(pg, float).copy()
    a = np.asarray(a, float)
    bz_s = np.asarray(bz_s, float)
    z_grip = float(pg[2])
    z_high = max(float(transit_z), z_grip + 0.10)
    p_slide = pg - float(slide) * bz_s - STANDOFF * a
    p_slide[2] = z_grip
    p_high = p_slide - float(out) * bz_s
    p_high[2] = z_high
    p_above = p_slide.copy()
    p_above[2] = z_high
    p_stand = pg - STANDOFF * a
    p_stand[2] = z_grip
    return [p_high, p_above, p_slide, p_stand, pg.copy()], p_slide


def _grasp_waypoints(ctx, pg, a, bz_s, slide, out, up, pull_scale=1.0):
    """High, board-away via → side entry → slide in. Returns (waypoints, p_slide)."""
    pg = np.asarray(pg, float)
    a = np.asarray(a, float)
    bz_s = np.asarray(bz_s, float)
    p_slide = pg - slide * bz_s - STANDOFF * a
    p_entry = p_slide - out * bz_s + np.array([0.0, 0.0, up])
    n = _board_away_normal(ctx)
    p_via = p_entry.copy()
    pull = 0.16 * float(pull_scale)
    bxy = _board_xy_prior(ctx)
    if bxy is not None:
        dist = float(np.linalg.norm(pg[:2] - bxy))
        pull = (0.22 if dist < 0.40 else 0.14) * float(pull_scale)
    if n is not None and pull > 1e-6:
        p_via = p_via - pull * n
    p_via[2] = max(float(p_via[2]), float(pg[2]) + 0.26, 0.34)
    p_via = _clip_to_workspace(ctx, p_via)
    p_entry = _clip_to_workspace(ctx, p_entry)
    # Keep entry above the housing so the via→entry chord clears the drill.
    p_entry[2] = max(float(p_entry[2]), float(pg[2]) + 0.10)
    p_entry = _clip_to_workspace(ctx, p_entry)
    wps = [p_via, p_entry, p_slide]
    wps += [pg - STANDOFF * a - s * slide * bz_s for s in (0.66, 0.33, 0.0)]
    wps.append(pg)
    return wps, p_slide


def _densify(wps, step=0.03):
    out = []
    wps = [np.asarray(p, float) for p in wps]
    for a, b in zip(wps[:-1], wps[1:]):
        L = float(np.linalg.norm(b - a))
        nseg = max(1, int(np.ceil(L / max(step, 1e-6))))
        for k in range(nseg):
            out.append(a + (k / nseg) * (b - a))
    out.append(wps[-1])
    return out


def _clip_to_workspace(ctx, p, margin=0.005):
    """Keep a via/entry inside the box; do not drop the whole grasp option."""
    g = ctx.gates
    lo = np.array([g.WS_X[0], g.WS_Y[0], g.WS_Z[0]], float) + margin
    hi = np.array([g.WS_X[1], g.WS_Y[1], g.WS_Z[1]], float) - margin
    return np.minimum(np.maximum(np.asarray(p, float), lo), hi)


def _gate_workspace(ctx, wps):
    for i, p in enumerate(wps):
        ctx.gates.check_workspace(p, f"grasp-wp{i}")


def _cart_poses(wps, R):
    R = np.asarray(R, float)
    return [(p, R) for p in _densify(wps)]


def _joint_poses(robot, chain, n_per=8, from_current=True):
    fk = getattr(robot, "fk_site", None)
    if fk is None or not chain:
        return []
    qs = []
    if from_current:
        qs.append(np.asarray(robot.get_q(), float))
    qs.extend(np.asarray(q, float) for q in chain)
    if len(qs) < 2:
        return [fk(qs[0])] if qs else []
    poses = []
    for a, b in zip(qs[:-1], qs[1:]):
        jump = float(np.linalg.norm(b - a))
        steps = max(n_per, int(np.ceil(jump / 0.08)))
        for k in range(steps):
            s = k / steps
            poses.append(fk(a + s * (b - a)))
    poses.append(fk(qs[-1]))
    return poses


def _ik_candidates(robot, wps, R, min_margin=0.2, max_jump=1.0):
    fn = getattr(robot, "plan_path_candidates", None)
    if fn is not None:
        return fn(wps, R, min_margin=min_margin, max_jump=max_jump)
    chain, margin = robot.plan_path_ik(wps, R, min_margin=min_margin, max_jump=max_jump)
    return [(0.0, chain, margin)]


def _chain_final_err(robot, chain, pg):
    """TCP distance at the last IK waypoint vs the grasp pose (m)."""
    fk = getattr(robot, "fk_site", None)
    if fk is None or not chain:
        return 0.0
    p, _ = fk(chain[-1])
    return float(np.linalg.norm(np.asarray(p, float) - np.asarray(pg, float)))


def _poses_hit_drill(poses, grip_xy, pg=None, approach_frac=0.40, radius=0.040, z_max=0.13):
    """True if early path samples (first ~40%) sweep through the drill housing."""
    del pg  # API compat; fraction of the polyline is the near-grasp window.
    grip_xy = np.asarray(grip_xy, float).reshape(2)
    n = len(poses)
    if n < 3:
        return False
    cut = max(1, int(n * float(approach_frac)))
    for p, _ in poses[:cut]:
        pw = np.asarray(p, float).reshape(3)
        if float(pw[2]) > z_max:
            continue
        if float(np.linalg.norm(pw[:2] - grip_xy)) < radius:
            return True
    return False


def _list_emergency_raised_paths(ctx, robot, pg, a, bz, Rg, side_order, obs, limit=6):
    """Last-resort 2-waypoint plans: high side hover → raised grip (seed 9).

    Skips the dense polyline that often has no continuous IK when the board is
    near the drill; execution still uses the raised-grip Cartesian drop.
    """
    gates = ctx.gates
    pg = np.asarray(pg, float)
    a = np.asarray(a, float)
    bz = np.asarray(bz, float)
    out = []
    opts = [(0.16, 0.10, 0.22), (0.20, 0.12, 0.26), (0.12, 0.08, 0.18),
            (0.18, 0.14, 0.28)]
    for side_sign in side_order:
        bz_s = float(side_sign) * bz
        for slide, outward, up in opts:
            p_slide = pg - slide * bz_s - STANDOFF * a
            p_hi = p_slide - outward * bz_s + np.array([0.0, 0.0, up])
            p_hi[2] = max(float(p_hi[2]), float(pg[2]) + 0.22, 0.32)
            p_hi = _clip_to_workspace(ctx, p_hi)
            # Duplicate hover as via+entry so execute's cart/raised fallback runs.
            wps = [p_hi, p_hi.copy(), p_slide, pg]
            try:
                _gate_workspace(ctx, wps)
                gates.check_workspace(pg, "emerg-grip")
            except GateError:
                continue
            cart = _cart_poses([p_hi, p_slide, pg], Rg)
            if obs is not None and poses_hit(cart, Rg, obs, BODY_SAMPLES):
                continue
            try:
                sols = list(robot.ik_solutions(p_hi, Rg, n_random=20))
            except Exception:
                continue
            for qv in sols[:3]:
                margin = float(robot.limit_margin(qv))
                try:
                    gates.check_joint_margin(margin)
                except GateError:
                    continue
                chain = [qv, qv]
                clr = 1.0 if obs is None else poses_clearance(cart, Rg, obs, BODY_SAMPLES)
                out.append((slide, outward, up, float(side_sign),
                            wps, chain, margin, p_slide, clr * 1000.0))
                if len(out) >= int(limit):
                    return out
                break
    return out


def _list_clear_grasp_paths(ctx, robot, pg, a, bz, Rg, options, side_order, obs,
                            check_drill=True, min_margin=0.2, max_final=0.030,
                            max_slide_err=0.050, limit=10):
    """Ranked LiDAR-clear IK chains (best clearance first)."""
    grip_xy = np.asarray(pg[:2], float)
    ranked = []
    for side_sign in side_order:
        bz_s = side_sign * bz
        for slide, out, up in options:
            wps, p_slide = _grasp_waypoints(ctx, pg, a, bz_s, slide, out, up)
            try:
                _gate_workspace(ctx, wps)
            except GateError as ex:
                ctx.log(f"approach option side={side_sign:+.0f} {slide, out, up} rejected: {ex}")
                continue
            cart = _cart_poses(wps, Rg)
            if obs is not None and poses_hit(cart, Rg, obs, BODY_SAMPLES):
                ctx.log(f"approach option side={side_sign:+.0f} {slide, out, up} "
                        "rejected: lidar board on Cartesian path")
                continue
            if check_drill and _poses_hit_drill(cart, grip_xy, pg):
                ctx.log(f"approach option side={side_sign:+.0f} {slide, out, up} "
                        "rejected: path through drill")
                continue
            try:
                cands = _ik_candidates(robot, wps, Rg, min_margin=min_margin)
            except (RuntimeError, GateError) as ex:
                ctx.log(f"approach option side={side_sign:+.0f} {slide, out, up} rejected: {ex}")
                continue
            for cost, chain, margin in cands:
                try:
                    ctx.gates.check_joint_margin(margin)
                except GateError:
                    continue
                ferr = _chain_final_err(robot, chain, pg)
                if ferr > float(max_final):
                    continue
                if len(chain) >= 3:
                    serr = _chain_final_err(robot, chain[:3], p_slide)
                    if serr > float(max_slide_err):
                        continue
                # Board: include current→via swing. Drill: only along the chain
                # (current→opposite-side via often chords through the housing).
                joint_board = _joint_poses(robot, chain, from_current=True)
                joint_drill = _joint_poses(robot, chain, from_current=False)
                check_board = joint_board if joint_board else cart
                check_drill_path = joint_drill if joint_drill else cart
                if obs is not None and poses_hit(check_board, Rg, obs, BODY_SAMPLES):
                    continue
                if check_drill and _poses_hit_drill(check_drill_path, grip_xy, pg):
                    continue
                clr = 1.0 if obs is None else poses_clearance(check_board, Rg, obs, BODY_SAMPLES)
                # Prefer short joint hop to the via from the current (nest) pose.
                jump0 = float(np.linalg.norm(
                    np.asarray(chain[0], float) - np.asarray(robot.get_q(), float)))
                rec = (clr, -jump0, -cost, -ferr, slide, out, up, side_sign, wps,
                       chain, margin, p_slide, clr * 1000.0)
                ranked.append(rec)
    ranked.sort(key=lambda r: r[:4], reverse=True)
    # Diversity: best per side, then longest slide per side, then fill by score.
    out = []
    seen_key = set()

    def _key(rec):
        return (round(float(rec[7]), 0), round(float(rec[4]), 2), round(float(rec[5]), 2))

    def _add(rec):
        key = _key(rec)
        if key in seen_key:
            return False
        seen_key.add(key)
        out.append(rec[4:])
        return True

    seen_side = set()
    for rec in ranked:
        side = round(float(rec[7]), 0)
        if side in seen_side:
            continue
        if _add(rec):
            seen_side.add(side)
        if len(out) >= int(limit):
            return out
    # Prefer a long side-slide (≥16 cm) on each side when available.
    for side_want in (1.0, -1.0):
        long_opts = [r for r in ranked
                     if round(float(r[7]), 0) == side_want and float(r[4]) >= 0.16]
        for rec in long_opts:
            _add(rec)
            if len(out) >= int(limit):
                return out
            break
    for rec in ranked:
        _add(rec)
        if len(out) >= int(limit):
            break
    return out


def _pick_clear_grasp_path(ctx, robot, pg, a, bz, Rg, options, side_order, obs,
                           check_drill=True):
    """Best single LiDAR-clear IK chain (compat wrapper)."""
    cands = _list_clear_grasp_paths(
        ctx, robot, pg, a, bz, Rg, options, side_order, obs, check_drill=check_drill)
    return cands[0] if cands else None
