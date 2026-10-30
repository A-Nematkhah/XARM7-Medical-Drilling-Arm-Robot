"""Sim-only drill kinematics + collision / force guards (callbacks on Robot.tick)."""
import os

import mujoco
import numpy as np
from calibration import on_board_face
from safety import GateError

from cutting import apply_cutting
from .constants import INDEX_TRIGGER_ON, MOTOR_SPEED
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
        idx = index_angle(ctx.hand)
        phys = trigger_physically_engaged(ctx)
        firm = trigger_pressed(ctx.hand)  # index <= INDEX_TRIGGER_ON
        soft = phys and idx <= INDEX_TRIGGER_ON + 80
        # Firm index close is enough (pad contact sometimes never moves trig qpos).
        # Soft band still needs physical travel so a half-closed finger cannot latch.
        if not st["motor"] and (firm or soft):
            st["motor"] = True
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
            on_face = on_board_face(hit, hole, n, ctx.calib.board.half)
            if on_face:
                # Phase 5: dry push aborts — do not ghost through without spinning.
                ctx.gates.check_dry_cut(pen_est, st["motor"], st["phase"])
            bit_dir = np.asarray(st.get("drill_dir", n), float)
            loads = apply_cutting(
                model, data, tip=tip, bit_dir=bit_dir, pen_m=pen_est if on_face else 0.0,
                motor_on=bool(st["motor"]), cfg=ctx.calib.cutting, tip_site=ctx.tip_site)
            st["cut_force_now"] = float(loads["force_n"])
            st["chuck_rpm"] = loads["rpm"]
            if on_face and st["motor"]:
                ctx.gates.check_cut_spin(loads["omega"], pen_est, st["phase"])
                st["cut_force_n"] = max(float(st.get("cut_force_n") or 0.0), loads["force_n"])
                st["cut_torque_nm"] = max(float(st.get("cut_torque_nm") or 0.0), abs(loads["torque_nm"]))
            _paint_viewer_hole()
        else:
            apply_cutting(
                model, data, tip=np.zeros(3), bit_dir=np.array([1.0, 0.0, 0.0]),
                pen_m=0.0, motor_on=False, cfg=ctx.calib.cutting, tip_site=-1)

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
        ph = str(st["phase"])
        # Grasp approach (3/8) can skim the board when the drill is nearby; abort on
        # real MuJoCo contact even before holding. Geometry held-check starts after grasp.
        if st["holding"]:
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
        elif not ph.startswith("3/"):
            return

        # During feed the chuck may rest on the face; the force gate is the abort.
        if ph.startswith(("7/", "FAILED")):
            return
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


def attach_hw_callbacks(ctx):
    """Hardware-loop motor latch + geometry gates. No MuJoCo cutting wrench.

    Attach on `--backend real`. Motor ON follows the index (the physical switch).
    `check_cut_spin` runs only when a chuck-speed reading is present.
    """
    st = ctx.st

    def motor_and_gates():
        idx = index_angle(ctx.hand)
        phys = trigger_physically_engaged(ctx)
        firm = trigger_pressed(ctx.hand)
        soft = phys and idx <= INDEX_TRIGGER_ON + 80
        if not st["motor"] and (firm or soft):
            st["motor"] = True
            ctx.log(f"trigger pulled (index {idx}) -> motor ON")
            ctx.runlog.event(st["phase"], "motor", on=True, index=idx)
        elif st["motor"] and trigger_released(ctx.hand):
            st["motor"] = False
            ctx.log("trigger released -> motor OFF")
            ctx.runlog.event(st["phase"], "motor", on=False)

        if not st["holding"] or ctx.tip_in_hand is None:
            return
        ph = str(st["phase"])
        p, R = ctx.robot.site_pose()
        if not ph.startswith(("7/", "FAILED")):
            ctx.gates.check_held_clearance(
                p, R, ctx.tip_in_hand, st["hole"], st["board_n"], ph)
            ctx.gates.check_transit_tip(
                measured_tip(ctx), st["hole"], st["board_n"], ph,
                board_half=ctx.calib.board.half)
        pen = estimated_penetration(ctx)
        ctx.gates.check_dry_cut(pen, st["motor"], ph)
        rpm = st.get("chuck_rpm")
        if rpm is not None and st["motor"] and pen > ctx.gates.MAX_DRY_PEN:
            omega = float(rpm) * 2.0 * np.pi / 60.0
            ctx.gates.check_cut_spin(omega, pen, ph)

    ctx.robot.callbacks.append(motor_and_gates)
