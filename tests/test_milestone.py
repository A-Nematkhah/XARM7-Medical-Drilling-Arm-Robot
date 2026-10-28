import numpy as np
import pytest

from safety import GateError, SafetyGates


N = np.array([1.0, 0.0, 0.0])


def test_perpendicular_passes_when_locked():
    g = SafetyGates()
    g.check_drill_angles(0, 0, N, N, allow_angled=False)


def test_angled_blocked_until_unlocked():
    g = SafetyGates()
    with pytest.raises(GateError, match="angled drilling locked"):
        g.check_drill_angles(15, 10, N, N, allow_angled=False)


def test_angled_allowed_when_unlocked():
    g = SafetyGates()
    # direction roughly into the board; angle magnitude still within sim limits
    d = np.array([0.95, 0.2, 0.24]); d /= np.linalg.norm(d)
    g.check_drill_angles(15, 10, d, N, allow_angled=True)
