"""Closed-loop drill feed: penetration rate tracks remaining depth.

`v_n ≈ kp * (target - pen) + ki * ∫err` along the board normal, then converted
to a bit-axis step. Speed is capped by `vmax` (gates / motion.yaml) and shrinks
as the hole approaches the target so the last millimetres do not overshoot.

Thrust and chuck RPM further derate `vmax` so a heavy cut does not stall the
motor. Missing sensors (`force=0`, `rpm=None`) leave that axis at 1.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DepthFeedCtrl:
    target: float
    vmax: float
    kp: float = 0.55
    ki: float = 0.10
    vmin: float = 0.0025
    max_step: float = 0.004
    tol: float = 0.001
    i_limit: float = 0.006
    dn: float = 1.0
    i_err: float = 0.0
    f_nom: float = 0.0
    f_hi: float = 0.0
    rpm_nom: float = 0.0
    rpm_lo: float = 0.0
    load_scale_min: float = 0.25

    def load_scale(self, force_n: float = 0.0, rpm: float | None = None,
                   motor_on: bool = False) -> float:
        """vmax multiplier in `[load_scale_min, 1]` from thrust and chuck speed.

        `rpm is None` is a missing tach (no RPM derate), even if the motor is on.
        `rpm=0` with the motor on is a stall and derates to the floor.
        """
        s = 1.0
        f = float(force_n)
        f_nom, f_hi = float(self.f_nom), float(self.f_hi)
        if f_hi > f_nom > 0.0 and f > f_nom:
            s = min(s, max(0.0, 1.0 - (f - f_nom) / (f_hi - f_nom)))
        if rpm is None:
            return max(float(self.load_scale_min), min(1.0, s))
        w = float(rpm)
        w_nom, w_lo = float(self.rpm_nom), float(self.rpm_lo)
        if w_nom > w_lo >= 0.0 and (motor_on or w > 0.0):
            s = min(s, max(0.0, (w - w_lo) / (w_nom - w_lo)))
        return max(float(self.load_scale_min), min(1.0, s))

    def command(self, pen_m: float, remaining_along: float, dt: float,
                force_n: float = 0.0, rpm: float | None = None, motor_on: bool = False):
        """Return `(step_along_m, speed_m_s)`. `(0, 0)` means stop (at target)."""
        err = float(self.target) - float(pen_m)
        if err <= self.tol:
            return 0.0, 0.0
        dn = max(float(self.dn), 1e-6)
        dt = max(float(dt), 1e-4)
        self.i_err = max(-self.i_limit, min(self.i_limit, self.i_err + err * dt))
        v_n = self.kp * err + self.ki * self.i_err
        v_along = v_n / dn
        scale = self.load_scale(force_n, rpm, motor_on=motor_on)
        vmax = float(self.vmax) * scale
        # Keep a floor speed until we are close, then allow a crawl to settle.
        # Load derate applies to the floor too so we do not push through a stall.
        vmin = (self.vmin * scale) if err > 3.0 * self.tol else 0.0
        v_along = max(vmin, min(vmax, v_along))
        need = (err + self.tol) / dn
        step = min(float(self.max_step), need, max(0.0, float(remaining_along)),
                   v_along * dt)
        if step <= 1e-5:
            return 0.0, v_along
        return float(step), float(v_along)

    @classmethod
    def from_profile(cls, prof, *, target, vmax, dn, tol):
        def _f(name, default):
            v = getattr(prof, name, None)
            return float(default) if v is None else float(v)
        return cls(
            target=float(target),
            vmax=float(vmax),
            kp=_f("kp", 0.55),
            ki=_f("ki", 0.10),
            vmin=_f("vmin", 0.0025),
            max_step=_f("max_step", 0.004),
            tol=float(tol),
            dn=float(dn),
            f_nom=_f("f_nom", 0.0),
            f_hi=_f("f_hi", 0.0),
            rpm_nom=_f("rpm_nom", 0.0),
            rpm_lo=_f("rpm_lo", 0.0),
            load_scale_min=_f("load_scale_min", 0.25),
        )
