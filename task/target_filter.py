"""EMA filter for board pierce point + face normal.

Raw marker detections jitter a few millimetres frame-to-frame. Approach used to
snap to each new hit (or hard-reject jumps > MAX_HOLE_JUMP). This filter:
  * rejects teleports larger than `max_jump`
  * blends accepted hole / normal updates toward the measurement
  * re-projects the hole onto the current face so depth noise cannot pull aim off-plane
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TargetFilter:
    hole: np.ndarray | None = None
    normal: np.ndarray | None = None
    alpha: float = 0.35          # weight of new hole measurement
    n_alpha: float = 0.20        # weight of new normal measurement
    max_jump: float = 0.05       # m; same role as MAX_HOLE_JUMP
    min_n_dot: float = 0.97      # refuse normals that flip / diverge
    n_updates: int = 0
    n_rejects: int = 0

    def seeded(self) -> bool:
        return self.hole is not None

    def seed(self, hole, normal=None):
        """Hard-set state (far look / first lock). Returns filtered hole."""
        self.hole = np.asarray(hole, float).reshape(3).copy()
        if normal is not None:
            n = np.asarray(normal, float).reshape(3)
            nrm = float(np.linalg.norm(n))
            self.normal = n / nrm if nrm > 1e-12 else None
        self.n_updates = 0
        self.n_rejects = 0
        return self.hole.copy()

    def update_hole(self, meas, *, plane_point=None, plane_n=None):
        """Blend `meas` into the filtered hole.

        Returns `(hole_or_None, jump_m, accepted)`.
        """
        meas = np.asarray(meas, float).reshape(3)
        if self.hole is None:
            self.seed(meas, plane_n if plane_n is not None else self.normal)
            if plane_point is not None and self.normal is not None:
                self.hole = _project_to_plane(self.hole, plane_point, self.normal)
            return self.hole.copy(), 0.0, True

        jump = float(np.linalg.norm(meas - self.hole))
        if jump > float(self.max_jump):
            self.n_rejects += 1
            return None, jump, False

        a = float(np.clip(self.alpha, 0.0, 1.0))
        blended = (1.0 - a) * self.hole + a * meas
        n = plane_n if plane_n is not None else self.normal
        if plane_point is not None and n is not None:
            blended = _project_to_plane(blended, plane_point, n)
        self.hole = np.asarray(blended, float).reshape(3)
        self.n_updates += 1
        return self.hole.copy(), jump, True

    def update_normal(self, meas_n):
        """Blend face normal; refuse flips. Returns filtered normal or None if rejected."""
        meas = np.asarray(meas_n, float).reshape(3)
        nrm = float(np.linalg.norm(meas))
        if nrm < 1e-12:
            return None
        meas = meas / nrm
        if self.normal is None:
            self.normal = meas.copy()
            return self.normal.copy()
        if float(meas @ self.normal) < float(self.min_n_dot):
            self.n_rejects += 1
            return None
        a = float(np.clip(self.n_alpha, 0.0, 1.0))
        blended = (1.0 - a) * self.normal + a * meas
        blended = blended / (np.linalg.norm(blended) + 1e-12)
        self.normal = blended
        return self.normal.copy()


def _project_to_plane(p, point, normal):
    n = np.asarray(normal, float).reshape(3)
    n = n / (np.linalg.norm(n) + 1e-12)
    p = np.asarray(p, float).reshape(3)
    c = np.asarray(point, float).reshape(3)
    return p - n * float((p - c) @ n)
