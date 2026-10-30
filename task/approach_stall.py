"""Approach-phase abort that is stricter than SafetyGates, without editing safety.py."""
from __future__ import annotations

from safety import GateError


class ApproachStall(GateError):
    """Cartesian servo is saturated against contact or a kinematic limit."""
