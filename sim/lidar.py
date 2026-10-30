"""Wrist 360° LiDAR: coarse drill/board XY so the D435i can look there.

Rays are physics (`mj_ray`). Live origin is TCP FK × `lidar_eye`, not a MuJoCo
site pose. Hits on the robot itself are dropped. Mark / pierce stay on the wrist
D435i.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calibration import camera_pose_from_tcp

try:
    import mujoco
except ImportError:  # pragma: no cover — real backend has no MuJoCo
    mujoco = None


@dataclass
class LidarScan:
    """World points from one 360° sweep. Empty rows are invalid."""
    origin: np.ndarray
    points: np.ndarray          # (N, 3), NaN where no return
    ranges: np.ndarray          # (N,) metres, inf if miss
    n_azim: int = 180
    n_elev: int = 5


@dataclass
class CoarseMap:
    """Approximate object poses for view planning. Not a pierce target."""
    drill_xy: np.ndarray | None = None
    board_xy: np.ndarray | None = None
    board_n: np.ndarray | None = None   # into-board, unit, mostly XY
    board_pts: np.ndarray | None = None  # world LiDAR returns on the wall
    n_points: int = 0


class WristLidar:
    """Multi-ring 360° scan from a puck on the hand (TCP × lidar_eye)."""

    # Steeper downward rings: the wrist sits higher than a column mast.
    ELEV_DEG = (-70.0, -48.0, -30.0, -14.0, 4.0)
    N_AZIM = 180
    MAX_RANGE = 1.6
    MIN_RANGE = 0.06

    def __init__(self, model, data, *, lidar_eye, tcp_pose_fn, noise=True, seed=0,
                 robot_root="link_base", exclude_body="wrist_lidar"):
        self.m, self.d = model, data
        self.lidar_eye = lidar_eye
        self.tcp_pose_fn = tcp_pose_fn
        self.noise = bool(noise)
        self.rng = np.random.default_rng(seed)
        self.robot_root = int(model.body(robot_root).id)
        try:
            self.exclude_body = int(model.body(exclude_body).id)
        except (KeyError, ValueError):
            self.exclude_body = -1
        self._robot = _descendant_ids(model, self.robot_root)
        self.last_scan: LidarScan | None = None
        self.last_map: CoarseMap | None = None

    def pose(self):
        p_ee, R_ee = self.tcp_pose_fn()
        return camera_pose_from_tcp(p_ee, R_ee, self.lidar_eye)

    def scan(self) -> LidarScan:
        origin, R = self.pose()
        az = np.linspace(0.0, 2 * np.pi, self.N_AZIM, endpoint=False)
        els = np.deg2rad(self.ELEV_DEG)
        dirs = []
        for el in els:
            ce, se = np.cos(el), np.sin(el)
            dirs.append(np.column_stack([ce * np.cos(az), ce * np.sin(az), np.full(az.shape, se)]))
        dir_l = np.concatenate(dirs, axis=0)
        dir_w = dir_l @ R.T
        n = len(dir_w)
        pts = np.full((n, 3), np.nan)
        rngs = np.full(n, np.inf)
        gid = np.zeros(1, np.int32)
        origin = np.asarray(origin, float)
        for i, v in enumerate(dir_w):
            # Skip self-hits (hand/arm in the near field) and keep casting — a
            # real 360° puck filters body returns; without this, pinky-side rays
            # never see a drill beside the palm.
            p = origin.copy()
            traveled = 0.0
            hit = None
            for _ in range(10):
                rem = self.MAX_RANGE - traveled
                if rem < self.MIN_RANGE:
                    break
                dist = float(mujoco.mj_ray(
                    self.m, self.d, p, v, None, 1, self.exclude_body, gid))
                if dist < 0.0 or dist > rem:
                    break
                bid = int(self.m.geom_bodyid[int(gid[0])])
                cand = p + dist * v
                traveled += dist
                if bid in self._robot or cand[2] < 0.025:
                    p = cand + 1e-3 * v
                    traveled += 1e-3
                    continue
                if traveled < self.MIN_RANGE:
                    p = cand + 1e-3 * v
                    traveled += 1e-3
                    continue
                hit = cand
                break
            if hit is None:
                continue
            if self.noise:
                traveled = float(traveled + self.rng.normal(0.0, 0.008))
                hit = origin + traveled * v
            pts[i] = hit
            rngs[i] = traveled
        scan = LidarScan(origin=origin.copy(), points=pts, ranges=rngs,
                         n_azim=self.N_AZIM, n_elev=len(self.ELEV_DEG))
        self.last_scan = scan
        return scan

    def locate(self, scan: LidarScan | None = None) -> CoarseMap:
        scan = self.scan() if scan is None else scan
        cmap = locate_objects(scan.points, origin_xy=scan.origin[:2])
        self.last_map = cmap
        return cmap


# Older name used while the puck lived on the column.
ColumnLidar = WristLidar


def _descendant_ids(model, root):
    ids = set()
    for b in range(model.nbody):
        x = b
        while x > 0:
            if x == root:
                ids.add(b)
                break
            x = int(model.body_parentid[x])
    ids.add(root)
    return ids


def locate_objects(points, origin_xy=(0.0, 0.0), eps=0.065, min_pts=4) -> CoarseMap:
    """Split the cloud into a table-top drill blob vs a tall vertical board wall."""
    P = np.asarray(points, float).reshape(-1, 3)
    ok = np.isfinite(P).all(axis=1)
    P = P[ok]
    origin_xy = np.asarray(origin_xy, float).reshape(2)
    if len(P) < min_pts:
        return CoarseMap(n_points=0)
    clusters = _cluster_xy(P, eps=eps, min_pts=min_pts)
    if not clusters:
        return CoarseMap(n_points=len(P))
    # Chuck / housing often split into two nearby blobs — fuse them first.
    clusters = _merge_nearby_flat(clusters, max_dist=0.22)
    scored = [_score_cluster(c, origin_xy) for c in clusters]
    board = max(scored, key=lambda s: s["board"])
    if board["board"] < 0.18:
        board = None
    drill_cands = [
        s for s in scored
        if s is not board
        and s["drill"] >= 0.10
        and s["z"] <= 0.32
        and not (s["z_span"] > 0.22 and s["span"] > 0.22)
    ]
    drill = None
    if drill_cands:
        drill = max(drill_cands, key=lambda s: s["drill"])
        # If several look drill-like, prefer the one farther from the board wall.
        if board is not None and len(drill_cands) > 1:
            top = float(drill["drill"])
            near = [s for s in drill_cands if float(s["drill"]) >= 0.80 * top]
            if len(near) > 1:
                drill = max(near, key=lambda s: float(np.linalg.norm(s["xy"] - board["xy"])))
    cmap = CoarseMap(n_points=len(P))
    if drill is not None:
        # Merged housing+chuck: bias look-at toward the half farther from the board
        # wall when we have one; otherwise the cluster mean.
        cmap.drill_xy = _tool_anchor_xy(
            drill["pts"], origin_xy, drill["xy"],
            board_xy=None if board is None else board["xy"])
    if board is not None:
        cmap.board_xy = board["xy"]
        cmap.board_n = board["n"]
        cmap.board_pts = np.asarray(board["pts"], float)
    return cmap


def _tool_anchor_xy(C, origin_xy, fallback, board_xy=None):
    """Look-at for an elongated tool (housing + chuck). Prefer denser / board-away half."""
    C = np.asarray(C, float).reshape(-1, 3)
    if len(C) < 6:
        return np.asarray(fallback, float).reshape(2)
    xy = C[:, :2]
    c = xy.mean(0)
    Q = xy - c
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    t = Q @ Vt[0]
    left, right = xy[t <= 0], xy[t > 0]
    if len(left) < 3 or len(right) < 3:
        return c
    # Chuck often has more tip hits than the housing. Prefer the half away
    # from the board wall; else the denser half.
    if board_xy is not None:
        b = np.asarray(board_xy, float).reshape(2)
        dl = float(np.linalg.norm(left.mean(0) - b))
        dr = float(np.linalg.norm(right.mean(0) - b))
        return left.mean(0) if dl >= dr else right.mean(0)
    return left.mean(0) if len(left) >= len(right) else right.mean(0)


def _merge_nearby_flat(clusters, max_dist=0.22, max_z_span=0.14):
    """Merge compact table-top blobs that belong to one tool (housing + chuck)."""
    if len(clusters) < 2:
        return clusters
    meta = []
    for c in clusters:
        meta.append((c, c[:, :2].mean(0), float(np.ptp(c[:, 2]))))
    used = set()
    out = []
    for i, (ci, xyi, zi) in enumerate(meta):
        if i in used:
            continue
        group = [ci]
        used.add(i)
        if zi <= max_z_span:
            for j, (cj, xyj, zj) in enumerate(meta):
                if j in used or zj > max_z_span:
                    continue
                if float(np.linalg.norm(xyi - xyj)) <= max_dist:
                    group.append(cj)
                    used.add(j)
                    xyi = np.vstack(group)[:, :2].mean(0)
        out.append(np.vstack(group) if len(group) > 1 else ci)
    return out


def _cluster_xy(P, eps, min_pts):
    used = np.zeros(len(P), bool)
    clusters = []
    xy = P[:, :2]
    for i in range(len(P)):
        if used[i]:
            continue
        seed = [i]
        used[i] = True
        k = 0
        while k < len(seed):
            d = np.linalg.norm(xy - xy[seed[k]], axis=1)
            nbr = np.where((~used) & (d < eps))[0]
            used[nbr] = True
            seed.extend(nbr.tolist())
            k += 1
        if len(seed) >= min_pts:
            clusters.append(P[np.array(seed)])
    return clusters


def _score_cluster(C, origin_xy):
    """Board = tall, elongated wall. Drill = low, flat, compact blob on the table."""
    xy = C[:, :2]
    z = C[:, 2]
    c = xy.mean(0)
    z_mean = float(z.mean())
    z_span = float(np.ptp(z))
    Q = xy - c
    _, s, Vt = np.linalg.svd(Q, full_matrices=False)
    span = float(np.ptp(Q @ Vt[0]))
    thin = float(s[1] / max(s[0], 1e-9)) if len(s) > 1 else 1.0
    n2 = Vt[-1].copy()
    away = c - origin_xy
    if float(n2 @ away) < 0:
        n2 = -n2
    n = np.array([n2[0], n2[1], 0.0])
    ln = np.linalg.norm(n)
    n = n / ln if ln > 1e-9 else np.array([1.0, 0.0, 0.0])
    n_pts = len(C)

    # Vertical face: high, thick in Z, long and thin in XY.
    board = (
        (0.35 + 0.65 * min(span / 0.30, 1.0))
        * (0.25 + 0.75 * min(z_span / 0.25, 1.0))
        * (0.5 + 0.5 * min(max(z_mean - 0.15, 0.0) / 0.20, 1.0))
        * (0.4 + 0.6 * (1.0 - min(thin, 1.0)))
        * min(n_pts / 25.0, 1.0)
    )
    # Drill housing on the table: small vertical extent, not wall-tall, not huge.
    compact = float(np.exp(-((span - 0.12) / 0.16) ** 2))
    low_wall = float(np.exp(-(z_span / 0.08) ** 2))
    # Housing hits sit ~5–25 cm above the table.
    table_z = float(np.exp(-((z_mean - 0.16) / 0.16) ** 2))
    drill = (
        (0.35 + 0.65 * compact)
        * (0.35 + 0.65 * low_wall)
        * (0.35 + 0.65 * table_z)
        * min(n_pts / 8.0, 1.0)
        * (0.7 + 0.3 * (1.0 - min(thin, 1.0)))  # slight preference for elongated housing
    )
    # Strong penalty if it already looks like the board.
    if z_span > 0.15 and span > 0.22 and z_mean > 0.28:
        drill *= 0.05
    if z_span < 0.08 and z_mean < 0.26:
        board *= 0.15
    return dict(
        xy=c, z=z_mean, z_span=z_span, span=span, thin=thin, n=n,
        n_pts=n_pts, board=board, drill=drill, pts=C,
    )
