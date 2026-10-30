"""LiDAR board as a finite slab + cloud. Used to pick a collision-free grasp path.

Hardware and sim share this: no MuJoCo contacts, only the wrist 360° returns.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Extra TCP-frame spheres beyond SafetyGates.HAND_SAMPLES: LiDAR puck, wrist, palm.
BODY_SAMPLES = np.array([
    [0.00, 0.00, 0.00],
    [0.00, 0.00, 0.10],
    [0.00, 0.00, 0.14],
    [0.00, 0.05, 0.10],
    [0.00, -0.04, 0.08],
    [-0.06, 0.00, 0.05],
    [-0.10, 0.03, 0.03],
    [-0.02, -0.08, -0.10],  # wrist LiDAR puck (pinky side)
    [-0.12, 0.00, 0.00],    # wrist
    [-0.16, 0.00, 0.00],    # link6 / proximal wrist
    [-0.18, 0.03, -0.04],
    [-0.18, -0.04, -0.04],
    [-0.14, 0.00, -0.08],
    [0.02, 0.04, 0.12],
    [0.02, -0.04, 0.12],
    [-0.08, -0.06, 0.04],
])


@dataclass
class BoardObstacle:
    """Finite face from the LiDAR wall cluster (not the calib hole hint)."""
    origin: np.ndarray
    n: np.ndarray
    u: np.ndarray
    v: np.ndarray
    u_lim: tuple
    v_lim: tuple
    pts: np.ndarray
    margin: float = 0.030

    @classmethod
    def from_points(cls, pts, n=None, margin=0.030):
        P = np.asarray(pts, float).reshape(-1, 3)
        ok = np.isfinite(P).all(axis=1)
        P = P[ok]
        if len(P) < 3:
            return None
        origin = P.mean(0)
        if n is None:
            Q = P - origin
            _, _, Vt = np.linalg.svd(Q, full_matrices=False)
            n = Vt[-1].copy()
            if float(n[:2] @ origin[:2]) < 0:
                n = -n
        n = np.asarray(n, float)
        n = n / (np.linalg.norm(n) + 1e-12)
        tmp = np.array([0.0, 0.0, 1.0])
        u = tmp - n * float(tmp @ n)
        if np.linalg.norm(u) < 0.2:
            tmp = np.array([0.0, 1.0, 0.0])
            u = tmp - n * float(tmp @ n)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(n, u)
        v = v / (np.linalg.norm(v) + 1e-12)
        rel = P - origin
        du = rel @ u
        dv = rel @ v
        return cls(
            origin=origin, n=n, u=u, v=v,
            u_lim=(float(du.min()), float(du.max())),
            v_lim=(float(dv.min()), float(dv.max())),
            pts=P, margin=float(margin),
        )

    @classmethod
    def from_map(cls, cmap, margin=0.030):
        if cmap is None:
            return None
        pts = getattr(cmap, "board_pts", None)
        if pts is None or len(np.asarray(pts).reshape(-1, 3)) < 3:
            if getattr(cmap, "board_xy", None) is None:
                return None
            xy = np.asarray(cmap.board_xy, float).reshape(2)
            pts = np.array([[xy[0], xy[1], 0.34]])
        return cls.from_points(pts, n=getattr(cmap, "board_n", None), margin=margin)

    def _frame(self, world):
        W = np.asarray(world, float).reshape(-1, 3)
        rel = W - self.origin
        return rel @ self.n, rel @ self.u, rel @ self.v, W

    def _on_face(self, du, dv, pad):
        return (
            (du >= self.u_lim[0] - pad) & (du <= self.u_lim[1] + pad)
            & (dv >= self.v_lim[0] - pad) & (dv <= self.v_lim[1] + pad)
        )

    def min_clearance(self, world):
        """Metres in front of the occupied face (negative = into / through)."""
        depth, du, dv, _ = self._frame(world)
        on = self._on_face(du, dv, self.margin)
        if not np.any(on):
            return float(-np.min(depth)) if len(depth) else 1.0
        return float(np.min(-depth[on]))

    def hits(self, world, margin=None) -> bool:
        pad = float(self.margin if margin is None else margin)
        depth, du, dv, W = self._frame(world)
        if np.any(self._on_face(du, dv, pad) & (depth > -pad)):
            return True
        P = self.pts
        if len(P) == 0 or len(W) == 0:
            return False
        if len(P) > 400:
            P = P[:: max(1, len(P) // 400)]
        d = np.linalg.norm(W[:, None, :] - P[None, :, :], axis=2)
        return float(np.min(d)) < pad


def body_world(p, R, samples=None):
    samples = BODY_SAMPLES if samples is None else np.asarray(samples, float)
    p = np.asarray(p, float).reshape(3)
    return p + (np.asarray(R, float) @ samples.T).T


def poses_hit(poses, R, obs: BoardObstacle, samples=None) -> bool:
    if obs is None:
        return False
    R = np.asarray(R, float)
    for p, Rp in poses:
        RR = R if Rp is None else np.asarray(Rp, float)
        if obs.hits(body_world(p, RR, samples)):
            return True
    return False


def poses_clearance(poses, R, obs: BoardObstacle, samples=None) -> float:
    if obs is None or not poses:
        return 1.0
    R = np.asarray(R, float)
    return min(
        obs.min_clearance(body_world(p, R if Rp is None else Rp, samples))
        for p, Rp in poses)
