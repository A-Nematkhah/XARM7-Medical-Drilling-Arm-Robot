"""Hardware backends for the drill task.

Sim today, real xArm/RH56/D435i later. Task code should depend on the
interfaces in `hw.interfaces`, not on MuJoCo types directly. Live motion on
hardware must use the controller — never Python `servo_step` / `mj_jacSite`.

`make_sim_backend` is imported lazily so `sim.robot` can load `hw.defaults`
without a circular import.
"""
from .interfaces import ArmDriver, HandDriver, WristCamera
from .real_backend import RealBackendNotReady, make_real_backend

__all__ = [
    "ArmDriver",
    "HandDriver",
    "WristCamera",
    "RealBackendNotReady",
    "make_sim_backend",
    "make_real_backend",
]


def make_sim_backend(*args, **kwargs):
    from .sim_backend import make_sim_backend as _make
    return _make(*args, **kwargs)
