"""Cartesian / joint motion helpers for the sim ArmDriver.

Live hardware must not import this — xArm owns the trajectory. The task talks
to `ArmDriver.move_to_pose` / `move_along` / `move_joint_path` only.
"""
from __future__ import annotations

import numpy as np


def min_jerk(s):
    s = np.clip(float(s), 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def polyline_from_qs(qs, min_len=1e-4):
    """[(q0, q1, length), ...] skipping near-duplicates."""
    segs = []
    prev = np.asarray(qs[0], float)
    for q in qs[1:]:
        qn = np.asarray(q, float)
        L = float(np.linalg.norm(qn - prev))
        if L > min_len:
            segs.append((prev, qn, L))
            prev = qn
    return segs


def q_along_polyline(segs, s):
    """Interpolate joint vector at arc-length `s` along `polyline_from_qs` segments."""
    if not segs:
        raise ValueError("empty joint polyline")
    s = max(0.0, float(s))
    acc = 0.0
    for q0, q1, L in segs:
        if s <= acc + L:
            u = 0.0 if L < 1e-12 else (s - acc) / L
            return q0 + u * (q1 - q0)
        acc += L
    return segs[-1][1].copy()


# Quintic min-jerk s(τ)=10τ³-15τ⁴+6τ⁵ has max ṡ = 30τ²(1-τ)² = 1.875 at τ=0.5.
MIN_JERK_PEAK = 1.875


def duration_for_distance(dist_m, T_cmd, vmax):
    """Lower-bound T so *peak* min-jerk Cartesian speed stays under vmax."""
    vmin = MIN_JERK_PEAK * abs(float(dist_m)) / max(float(vmax), 1e-6)
    return max(float(T_cmd), float(vmin))


class CartesianPI:
    """Integral on TCP position error so DLS servo rejects a steady cutting load.

    `u = kp * e + ki * ∫e`. The IK step already maps `e` through J⁺; we add the
    integral as an extra position offset on the servo target. Anti-windup: freeze
    I when the joint setpoint is clamped to the measured-q windup band.
    """

    def __init__(self, dt, ki=0.35, i_limit=0.006):
        self.dt = float(dt)
        self.ki = float(ki)
        self.i_limit = float(i_limit)
        self.i_pos = np.zeros(3)

    def reset(self):
        self.i_pos[:] = 0.0

    def target(self, p_t):
        return np.asarray(p_t, float) + self.ki * self.i_pos

    def update(self, pos_err, saturated):
        if saturated:
            return
        self.i_pos = np.clip(
            self.i_pos + np.asarray(pos_err, float) * self.dt,
            -self.i_limit, self.i_limit)
