"""Sim cutting loads: axial thrust + chuck torque while the bit is in the board.

Bit geoms stay non-colliding so the flute can occupy the hole (MuJoCo cannot
remove wood). Material resistance is applied as a wrench on the drill / a
torque on `chuck_spin`. Hardware feels the real workpiece; do not port this
to `--backend real`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CuttingCalib:
    """Wood-like resistance. Tune so the velocity motor still spins at 25 mm depth."""
    enabled: bool = True
    axial_n_per_m: float = 240.0       # ~6 N thrust at 25 mm
    axial_damp_ns_per_m: float = 40.0  # extra N per (m/s) of feed into the face
    torque_nm_per_m: float = 4.0       # ~0.10 N*m at 25 mm (motor forcerange 0.5)
    viscous_nm_s: float = 0.0004       # extra chuck damping in material


def cutting_loads(pen_m, v_along, omega, motor_on, cfg: CuttingCalib):
    """Return (F_resist_N >= 0, T_spin_Nm signed to oppose omega).

    `v_along` is tip speed along the bit into the board (positive = feeding in).
    """
    if not cfg.enabled or not motor_on or float(pen_m) <= 0.0:
        return 0.0, 0.0
    pen = float(pen_m)
    v = max(0.0, float(v_along))
    w = float(omega)
    force = cfg.axial_n_per_m * pen + cfg.axial_damp_ns_per_m * v
    mag = cfg.torque_nm_per_m * pen + cfg.viscous_nm_s * abs(w)
    sign = -1.0 if w >= 0.0 else 1.0
    return float(force), float(sign * mag)


def _clear_wrench(model, data, drill_bid, chuck_dof):
    data.xfrc_applied[drill_bid] = 0.0
    if chuck_dof >= 0:
        data.qfrc_applied[chuck_dof] = 0.0


def apply_cutting(model, data, *, tip, bit_dir, pen_m, motor_on, cfg: CuttingCalib,
                  tip_site=-1):
    """Write cutting wrench into `data`. Returns a dict of loads (for logs).

    Force is applied at the tip (equivalent wrench on the drill body). Torque
    is applied on the `chuck_spin` hinge so the velocity motor has to fight it.
    """
    drill_bid = int(model.body("drill").id)
    jnt = int(model.joint("chuck_spin").id)
    chuck_dof = int(model.jnt_dofadr[jnt])
    if not cfg.enabled:
        _clear_wrench(model, data, drill_bid, chuck_dof)
        return dict(force_n=0.0, torque_nm=0.0, omega=0.0, rpm=0.0, v_along=0.0)

    d = np.asarray(bit_dir, float)
    nrm = float(np.linalg.norm(d))
    d = d / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0])
    omega = float(data.qvel[chuck_dof])

    v_along = 0.0
    if int(tip_site) >= 0:
        import mujoco
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, int(tip_site), vel, 0)
        v_along = float(np.asarray(vel[3:6]) @ d)

    force, torque = cutting_loads(pen_m, v_along, omega, motor_on, cfg)
    _clear_wrench(model, data, drill_bid, chuck_dof)
    if force > 0.0 or abs(torque) > 0.0:
        F = -force * d  # material pushes the bit back (toward the palm)
        com = np.asarray(data.xpos[drill_bid], float)
        r = np.asarray(tip, float) - com
        data.xfrc_applied[drill_bid, :3] = F
        data.xfrc_applied[drill_bid, 3:] = np.cross(r, F)
        data.qfrc_applied[chuck_dof] = torque
    return dict(
        force_n=round(force, 3),
        torque_nm=round(torque, 4),
        omega=round(omega, 3),
        rpm=round(abs(omega) * 60.0 / (2.0 * np.pi), 1),
        v_along=round(v_along, 4),
    )
