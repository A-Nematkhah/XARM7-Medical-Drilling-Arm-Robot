"""Close the RH56 until wrap fingers are loaded, then hold that squeeze.

Position-controlled hands keep preload by freezing the *command* that first
exceeded `gates.finger_load`, not the measured angle (that would unload).
Index stays open for the trigger. Missing force sensors time out and the
caller still runs `SafetyGates.check_grasp`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# little, ring, middle, thumb_bend — same set as SafetyGates.check_grasp
LOADED_IDX = (0, 1, 2, 4)
INDEX = 3


def overlay_index(cmd, index_angle):
    """Keep wrap-finger commands; only change the trigger (index) joint."""
    out = np.clip(np.asarray(cmd, float).reshape(-1)[:6].copy(), 0.0, 1000.0)
    if out.size < 6:
        out = np.pad(out, (0, 6 - out.size))
    out[INDEX] = float(np.clip(index_angle, 0.0, 1000.0))
    return out


WRAP_TARGET = [0, 0, 0, 1000, 1000, 0]
CLOSE_DT = 0.05
CLOSE_TIMEOUT = 3.5
CLOSE_SPEED = 800
CLOSE_SETTLE = 0.15
HOLD_SCALE = 1.25   # freeze past the gate so settle cannot drop below finger_load
HOLD_EXTRA = 100.0  # extra closed units once a finger trips; fights cut-slip


def n_loaded(forces, load, loaded_idx=LOADED_IDX) -> int:
    f = np.abs(np.asarray(forces, float).reshape(-1))
    idx = [i for i in loaded_idx if 0 <= i < f.size]
    return int(np.sum(f[idx] >= float(load)))


@dataclass
class GraspCloseResult:
    forces: np.ndarray
    n_loaded: int
    elapsed: float
    cmd: np.ndarray
    frozen: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=bool))


def close_until_load(
    hand, robot, *,
    target,
    load,
    min_loaded,
    loaded_idx=LOADED_IDX,
    dt=CLOSE_DT,
    timeout=CLOSE_TIMEOUT,
    speed=CLOSE_SPEED,
    settle=CLOSE_SETTLE,
    require_thumb=True,
    hold_scale=HOLD_SCALE,
    hold_extra=HOLD_EXTRA,
):
    """Step wrap fingers toward `target` until `min_loaded` of `loaded_idx` exceed `load`.

    Returns a `GraspCloseResult` even on timeout; the caller must `check_grasp`.
    """
    target = np.clip(np.asarray(target, float).reshape(-1), 0.0, 1000.0)
    if target.size != 6:
        raise ValueError("hand target must be 6 angles")
    loaded_idx = tuple(int(i) for i in loaded_idx)
    cmd = np.clip(np.asarray(hand.get_angles(), float).reshape(-1)[:6].copy(), 0.0, 1000.0)
    frozen = np.zeros(6, dtype=bool)
    step = max(10.0, float(speed) * float(dt))
    extra = max(0.0, float(hold_extra))
    hold = float(load) * max(1.0, float(hold_scale))
    elapsed = 0.0
    forces = np.abs(np.asarray(hand.get_forces(), float).reshape(-1))
    n = n_loaded(forces, hold, loaded_idx)

    def _thumb_ok():
        if not require_thumb or 4 not in loaded_idx:
            return True
        if frozen[4]:
            return True
        return forces.size > 4 and float(forces[4]) >= hold

    def _trip(i):
        if frozen[i]:
            return
        frozen[i] = True
        if extra > 0.0:
            if cmd[i] > target[i]:
                cmd[i] = max(float(target[i]), cmd[i] - extra)
            elif cmd[i] < target[i]:
                cmd[i] = min(float(target[i]), cmd[i] + extra)

    while elapsed < float(timeout) and not (n >= int(min_loaded) and _thumb_ok()):
        forces = np.abs(np.asarray(hand.get_forces(), float).reshape(-1))
        if forces.size < 6:
            forces = np.pad(forces, (0, 6 - forces.size))
        for i in loaded_idx:
            if frozen[i] or float(forces[i]) >= hold:
                _trip(i)
                continue
            if cmd[i] > target[i]:
                cmd[i] = max(float(target[i]), cmd[i] - step)
            elif cmd[i] < target[i]:
                cmd[i] = min(float(target[i]), cmd[i] + step)
        for i in range(6):
            if i not in loaded_idx:
                cmd[i] = float(target[i])
        hand.set_angles(cmd, speed=speed)
        robot.wait(dt)
        elapsed += float(dt)
        forces = np.abs(np.asarray(hand.get_forces(), float).reshape(-1))
        if forces.size < 6:
            forces = np.pad(forces, (0, 6 - forces.size))
        for i in loaded_idx:
            if float(forces[i]) >= hold:
                _trip(i)
        n = n_loaded(forces, hold, loaded_idx)
        if all(frozen[i] for i in loaded_idx):
            break

    hand.set_angles(cmd, speed=speed)
    if settle > 0.0:
        robot.wait(settle)
        elapsed += float(settle)
    forces = np.abs(np.asarray(hand.get_forces(), float).reshape(-1))
    n = n_loaded(forces, load, loaded_idx)
    return GraspCloseResult(
        forces=forces, n_loaded=n, elapsed=elapsed, cmd=cmd.copy(), frozen=frozen)
