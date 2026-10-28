"""Sim-only drill kinematics + collision / force guards (callbacks on Robot.tick)."""
import os

import mujoco
import numpy as np
from calibration import on_board_face
from safety import GateError

from .constants import MOTOR_SPEED
from .geom import quat_align_z
from .hand_io import index_angle, trigger_physically_engaged, trigger_pressed, trigger_released
from .tip import estimated_penetration, measured_tip


def attach_sim_callbacks(ctx):
    """MuJoCo-only motor / contact / hole-paint. Do not call on --backend real."""
    if ctx.model is None or ctx.data is None:
        return
    model, data, st = ctx.model, ctx.data, ctx.st
    motor_act = model.actuator("drill_motor").id
    hole_gid = model.geom("hole").id
    board_bid = model.body("board").id
    chuck_bid = model.body("chuck").id
    force6 = np.zeros(6)

    def drill_logic():
        # Motor follows the index command, but only if the physical trigger
        # has actually moved (sim). On hardware the index *is* the switch.
        pressed = trigger_pressed(ctx.hand) and trigger_physically_engaged(ctx)
        if not st["motor"] and pressed:
            st["motor"] = True
            idx = index_angle(ctx.hand)
            ctx.log(f"trigger pulled (index {idx}) -> motor ON")
            ctx.runlog.event(st["phase"], "motor", on=True, index=idx)
        elif st["motor"] and trigger_released(ctx.hand):
            st["motor"] = False
            ctx.log("trigger released -> motor OFF")
            ctx.runlog.event(st["phase"], "motor", on=False)
        data.ctrl[motor_act] = MOTOR_SPEED if st["motor"] else 0.0

        # Live depth is estimated tip vs the *detected* plane (same as hardware).
        if st["holding"] and ctx.tip_in_hand is not None:
            pen_est = estimated_penetration(ctx)
            if st["motor"] and pen_est > st["depth"]:
                st["depth"] = pen_est
            tip = measured_tip(ctx)
            hole, n = st["hole"], st["board_n"]
            hit = tip + ctx.gates.tip_clearance(tip, hole, n) * np.asarray(n, float)
            if on_board_face(hit, hole, n, ctx.calib.board.half):
                # Phase 5: dry push aborts — do not ghost through without spinning.
                ctx.gates.check_dry_cut(pen_est, st["motor"], st["phase"])
            _paint_viewer_hole()

    def _paint_viewer_hole():
        """Viewer disc only. Uses the detected plane and calibrated face size."""
        if not st["motor"] or ctx.tip_in_hand is None:
            return
        half = ctx.calib.board.half
        n = np.asarray(st["board_n"], float)
        d = np.asarray(st["drill_dir"], float)
        hole = np.asarray(st["hole"], float)
        dn = float(d @ n)
        if abs(dn) < 1e-6:
            return
        tip = measured_tip(ctx)
        clear = ctx.gates.tip_clearance(tip, hole, n)
        pen = -float(clear)
        if pen <= 0:
            return
        hit = np.asarray(tip, float) + float(clear) * n
        if not on_board_face(hit, hole, n, half):
            return
        max_pen = 2.0 * float(half[0]) / max(abs(dn), 0.2) + 0.005
        if pen >= max_pen:
            return
        bp = data.xpos[board_bid]
        Rm = data.xmat[board_bid].reshape(3, 3)
        rel = Rm.T @ (hit - bp)
        model.geom_pos[hole_gid] = [-float(half[0]) - 0.0005, rel[1], rel[2]]
        model.geom_quat[hole_gid] = quat_align_z(d)
        model.geom_size[hole_gid][:2] = [0.0045, 0.0006]
        model.geom_rgba[hole_gid][3] = 1.0

    def _board_contact_force():
        """Sum |F| for contacts involving the board (chuck/housing/hand)."""
        ncon = getattr(data, "ncon", 0) or 0
        total = 0.0
        for i in range(ncon):
            c = data.contact[i]
            b1 = int(model.geom_bodyid[c.geom1])
            b2 = int(model.geom_bodyid[c.geom2])
            if board_bid not in (b1, b2):
                continue
            mujoco.mj_contactForce(model, data, i, force6)
            total += float(np.linalg.norm(force6[:3]))
        return total

    def board_collision_guard():
        if not st["holding"]:
            return
        ph = str(st["phase"])
        # Geometry gate (works on hardware without ncon). Skip during feed while
        # the bit is intentionally past the face — housing samples still apply via
        # check_aim_clearance at approach; here we only watch non-drill phases.
        if (not ph.startswith(("7/", "FAILED"))) and ctx.tip_in_hand is not None:
            p, R = ctx.robot.site_pose()
            ctx.gates.check_held_clearance(
                p, R, ctx.tip_in_hand, st["hole"], st["board_n"], ph)

        # Phase 5: chuck↔board is real contact. Force gate always; body hit aborts
        # even when motor is on (over-feed into the chuck housing).
        force_n = _board_contact_force()
        if force_n > 0.0:
            ctx.gates.check_board_contact_force(force_n, ph)
        ncon = getattr(data, "ncon", None)
        if ncon is None:
            return
        for i in range(ncon):
            c = data.contact[i]
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            if board_bid not in (b1, b2):
                continue
            other = b2 if b1 == board_bid else b1
            # Bit geoms are contype=0; any remaining contact is chuck/housing/hand.
            raise GateError(
                f"collision: {model.body(other).name} hit the board ({st['phase']})")

    def tip_transit_guard():
        if not st["holding"]:
            return
        ph = str(st["phase"])
        if ph.startswith(("7/", "FAILED")):
            return
        ctx.gates.check_transit_tip(
            measured_tip(ctx), st["hole"], st["board_n"], st["phase"],
            board_half=ctx.calib.board.half)

    ctx.robot.callbacks.append(drill_logic)
    ctx.robot.callbacks.append(board_collision_guard)
    ctx.robot.callbacks.append(tip_transit_guard)

    if os.environ.get("DBG_CONTACT"):
        seen = set()

        def _contact_dbg():
            did = model.body("drill").id
            for c in data.contact[:data.ncon]:
                b1 = model.body_rootid[model.geom_bodyid[c.geom1]]
                b2 = model.body_rootid[model.geom_bodyid[c.geom2]]
                if did in (b1, b2) and 0 not in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]):
                    other = model.body(model.geom_bodyid[c.geom1 if b2 == did else c.geom2]).name
                    if other not in seen:
                        seen.add(other)
                        ctx.log(f"first contact drill <-> {other} ({st['phase']})")
                        ctx.runlog.event(st["phase"], "contact", other=other)

        ctx.robot.callbacks.append(_contact_dbg)
