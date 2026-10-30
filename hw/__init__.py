"""Hardware backends for the drill task.

Sim via `make_sim_backend`; real xArm/RH56/D435i via `make_real_backend`
(optional SDKs in `requirements-hw.txt`). Task code should depend on the
interfaces in `hw.interfaces`, not on MuJoCo types. Live motion on hardware
must use the controller — never Python `servo_step` / `mj_jacSite`.

`make_sim_backend` is imported lazily so `sim.robot` can load `hw.defaults`
without a circular import.
"""
from .interfaces import (
    ArmDriver, BaseCamera, ColumnLidarDriver, HandDriver, WristCamera, WristLidarDriver,
)
from .real_backend import RealBackendNotReady, make_real_backend

__all__ = [
    "ArmDriver",
    "HandDriver",
    "WristCamera",
    "BaseCamera",
    "WristLidarDriver",
    "ColumnLidarDriver",
    "RealBackendNotReady",
    "make_sim_backend",
    "make_real_backend",
]


def make_sim_backend(*args, **kwargs):
    from .sim_backend import make_sim_backend as _make
    return _make(*args, **kwargs)
