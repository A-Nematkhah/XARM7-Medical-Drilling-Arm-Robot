"""Bit tip from the grasp frame (TCP + tip_in_hand). No MuJoCo bit_tip site."""
import numpy as np


def bind_tip_in_hand(p_tcp, R_tcp, bit_tip_world):
    """Grasp-frame tip from the actual TCP and a perceived world tip.

    Call this at the grasp pose (actual `site_pose`, not the commanded `pg`)
    so tracking residual is absorbed into `tip_in_hand`.
    """
    p = np.asarray(p_tcp, float)
    R = np.asarray(R_tcp, float)
    return R.T @ (np.asarray(bit_tip_world, float) - p)


def measured_tip(ctx):
    """World position of the bit tip from TCP pose and the in-hand estimate.

    `tip_in_hand` is set at grasp from perception (or a calibrated tool
    transform on hardware). The live loop never reads the sim `bit_tip` site.
    """
    p, R = ctx.robot.site_pose()
    return p + R @ np.asarray(ctx.tip_in_hand, float)


def estimated_penetration(ctx, hole=None, n=None):
    """How far the estimated tip is past the detected face, along the normal."""
    hole = ctx.st["hole"] if hole is None else hole
    n = ctx.st["board_n"] if n is None else n
    clear = ctx.gates.tip_clearance(measured_tip(ctx), hole, n)
    return max(0.0, -float(clear))
