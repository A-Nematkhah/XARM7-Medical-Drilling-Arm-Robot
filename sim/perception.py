"""
Wrist-camera perception for the drill task.

D435i      : renders aligned RGB + depth from the MuJoCo camera 'd435i_color',
             adds a RealSense-like depth noise model, and deprojects pixels to
             world points (same math as rs2_deproject_pixel_to_point + extrinsics).
DrillDetector:
   1. segment the yellow housing (HSV) and the silver bit (low-S / high-V)
   2. deproject masked pixels with depth -> world point clouds
   3. housing top points  -> PCA -> drill axis (xy) ; bit points fix the sign
   4. housing radius / axis height -> underside of the housing
   5. grip points (below the housing) -> grip axis, corrected for the fact that
      a single view only sees the camera-facing half of a cylinder
   6. bit tip = housing-axis centre + CAD housing_to_tip (thin bit is not trusted)
Nothing here reads simulator ground truth; only rendered images + camera pose.
To use a real D435i, replace D435i.capture() with pyrealsense2 frames
(align depth to color) and keep the rest.
"""
from collections import deque
from dataclasses import dataclass, field
import numpy as np
import cv2
import mujoco
from calibration import board_axes, camera_pose_from_base, camera_pose_from_tcp, intersect_ray_plane


def contour_circularity(cnt):
    """4π area / perimeter² — 1.0 is a perfect circle."""
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    if peri < 1e-6 or area < 1.0:
        return 0.0, area
    return float(4.0 * np.pi * area / (peri * peri)), area


def fit_circle(pts, r_prior=None, r_bounds=(0.005, 0.08), trim=0.003, iters=3):
    """Circle fit with iterative outlier trimming (drops points > `trim` m off the circle)."""
    keep = np.ones(len(pts), bool)
    for _ in range(iters):
        c, r = _fit_circle(pts[keep], r_prior, r_bounds)
        res = np.abs(np.linalg.norm(pts - c, axis=1) - r)
        new = res < max(trim, np.percentile(res, 50))
        if new.sum() < 10 or np.array_equal(new, keep):
            break
        keep = new
    return c, r


def _fit_circle(pts, r_prior=None, r_bounds=(0.005, 0.08)):
    """Least-squares circle fit (Kasa) to 2D points on a partial arc.
    If the free-radius fit is implausible, refit with the radius fixed to r_prior."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    sol, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    c = sol[:2]; r = float(np.sqrt(max(sol[2] + c @ c, 1e-12)))
    if r_prior is not None and not (r_bounds[0] < r < r_bounds[1]):
        # fixed radius: Gauss-Newton on the centre only
        c = pts.mean(0)
        for _ in range(30):
            dv = pts - c; dn = np.linalg.norm(dv, axis=1) + 1e-9
            res = dn - r_prior; J = -dv / dn[:, None]
            c = c - np.linalg.lstsq(J, res, rcond=None)[0]
        r = r_prior
    return c, r


class D435i:
    def __init__(self, model, data, name="d435i_color", width=640, height=360,
                 noise=True, seed=0, min_z=0.105, max_z=3.0,
                 hand_eye=None, tcp_pose_fn=None,
                 base_eye=None, base_pose_fn=None,
                 latency_s=0.0, dropout=0.01, edge_noise_m=0.0):
        self.m, self.d = model, data
        self.cam = model.camera(name).id
        self.W, self.H = width, height
        fovy = np.deg2rad(model.cam_fovy[self.cam])
        self.fy = self.fx = 0.5 * height / np.tan(0.5 * fovy)
        self.cx, self.cy = (width - 1) / 2.0, (height - 1) / 2.0
        self.rgb_r = mujoco.Renderer(model, height, width)
        self.dep_r = mujoco.Renderer(model, height, width)
        self.dep_r.enable_depth_rendering()
        self.noise, self.rng = noise, np.random.default_rng(seed)
        self.min_z, self.max_z = min_z, max_z
        self.hand_eye = hand_eye
        self.tcp_pose_fn = tcp_pose_fn
        self.base_eye = base_eye
        self.base_pose_fn = base_pose_fn
        self.latency_s = float(latency_s)
        self.dropout = float(dropout)
        self.edge_noise_m = float(edge_noise_m)
        # Capture rate of the task loop is ~controller tick; approximate with 30 Hz.
        self._latency_frames = max(0, int(round(self.latency_s * 30.0)))
        self._frame_buf = deque(maxlen=self._latency_frames + 1)

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def flush(self):
        """Drop buffered frames (call after a large move before the next look)."""
        self._frame_buf.clear()

    def settle_capture(self):
        """Capture after motion: refill the delay line from the *current* pose.

        Without this, `capture()` can return a frame from mid-reorient and
        board/marker detection fails (USB/ISP latency still applies once filled).
        """
        self.flush()
        out = None
        for _ in range(max(1, self._latency_frames + 1)):
            out = self.capture()
        return out

    def capture(self):
        """Returns rgb (H,W,3 uint8), depth in metres (H,W float32, 0 = invalid).

        When `latency_s` > 0 the returned pair is an older capture (USB / ISP delay).
        """
        self.rgb_r.update_scene(self.d, camera=self.cam)
        rgb = self.rgb_r.render().copy()
        self.dep_r.update_scene(self.d, camera=self.cam)
        z = self.dep_r.render().astype(np.float32)
        if self.noise:
            # D435-like: error grows ~ z^2, 1 mm quantisation, sparse dropouts,
            # plus extra noise on depth edges (flying pixels).
            sigma = 0.0008 + 0.0019 * z ** 2
            z = z + self.rng.normal(0, 1, z.shape).astype(np.float32) * sigma
            if self.edge_noise_m > 0:
                gz = np.abs(cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)) + \
                     np.abs(cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3))
                edge = gz > 0.02
                z[edge] = z[edge] + self.rng.normal(
                    0, self.edge_noise_m, int(edge.sum())).astype(np.float32)
            z = np.round(z * 1000) / 1000
            drop = max(float(self.dropout), 0.0)
            if drop > 0.0:
                z[self.rng.random(z.shape) < drop] = 0
        z[(z < self.min_z) | (z > self.max_z)] = 0
        self._frame_buf.append((rgb, z))
        if len(self._frame_buf) <= self._latency_frames:
            # Warm-up: return the newest frame until the delay line is full.
            return rgb, z
        return self._frame_buf[0]

    def mujoco_pose(self):
        """Sim renderer pose (eval only). Live `pose()` uses TCP FK + hand-eye."""
        return self.d.cam_xpos[self.cam].copy(), self.d.cam_xmat[self.cam].reshape(3, 3).copy()

    def pose(self):
        """Camera pose in world. Optical frame: x right, y up, looks along -z."""
        if self.hand_eye is not None and self.tcp_pose_fn is not None:
            p_ee, R_ee = self.tcp_pose_fn()
            return camera_pose_from_tcp(p_ee, R_ee, self.hand_eye)
        if self.base_eye is not None:
            if self.base_pose_fn is not None:
                p_b, R_b = self.base_pose_fn()
            else:
                p_b, R_b = np.zeros(3), np.eye(3)
            return camera_pose_from_base(p_b, R_b, self.base_eye)
        return self.mujoco_pose()

    def deproject(self, us, vs, zs, cam_pose=None):
        p, R = self.pose() if cam_pose is None else cam_pose
        x = (us - self.cx) / self.fx * zs
        y = -(vs - self.cy) / self.fy * zs      # image v points down, camera y up
        pc = np.stack([x, y, -zs], axis=1)
        return pc @ R.T + p

    def pixel_ray(self, u, v, cam_pose=None):
        """World ray through pixel (u,v). Camera looks along -Z, +Y up."""
        p, R = self.pose() if cam_pose is None else cam_pose
        dir_cam = np.array([(u - self.cx) / self.fx, -(v - self.cy) / self.fy, -1.0])
        d = R @ dir_cam
        return p, d / (np.linalg.norm(d) + 1e-12)

    def ray_plane(self, u, v, center, normal, cam_pose=None):
        """Intersect the pixel ray with a plane (no depth required)."""
        return intersect_ray_plane(*self.pixel_ray(u, v, cam_pose), center, normal)

    def project(self, P, cam_pose=None):
        p, R = self.pose() if cam_pose is None else cam_pose
        pc = (np.atleast_2d(P) - p) @ R
        z = -pc[:, 2]
        return np.stack([self.fx * pc[:, 0] / z + self.cx, -self.fy * pc[:, 1] / z + self.cy], 1), z


@dataclass
class DrillEstimate:
    grip_xy: np.ndarray           # grip (handle) axis position on the table plane
    axis: np.ndarray              # unit vector, horizontal, pointing to the chuck
    housing_underside_z: float
    housing_axis_z: float
    bit_tip: np.ndarray           # world point
    grip_radius: float
    n_points: dict = field(default_factory=dict)

    @property
    def yaw(self):
        return float(np.arctan2(self.axis[1], self.axis[0]))


class DrillDetector:
    # HSV thresholds (OpenCV ranges H:0-180, S,V:0-255)
    YELLOW_LO, YELLOW_HI = (15, 170, 35), (35, 255, 255)
    SILVER_LO, SILVER_HI = (0, 0, 120), (180, 45, 255)
    BIT_REACH = 0.2075          # grip axis -> bit tip (grasp)
    HOUSING_TO_TIP = 0.2115     # housing-axis centre -> bit tip (live estimate)

    def __init__(self, cam: D435i, table_z=0.0, bit_reach=None, housing_to_tip=None):
        self.cam, self.table_z = cam, table_z
        self.bit_reach = float(self.BIT_REACH if bit_reach is None else bit_reach)
        self.housing_to_tip = float(
            self.HOUSING_TO_TIP if housing_to_tip is None else housing_to_tip)
        self.last_debug = None

    def _mask(self, rgb, lo, hi, open_k=3):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mk = cv2.inRange(hsv, np.array(lo), np.array(hi))
        if open_k:
            mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
        return mk

    def _cloud(self, mask, depth, cam_pose):
        vs, us = np.nonzero((mask > 0) & (depth > 0))
        if len(us) == 0:
            return np.zeros((0, 3))
        return self.cam.deproject(us.astype(float), vs.astype(float), depth[vs, us], cam_pose)

    def detect_coarse(self, rgb, depth, cam_pose):
        """First look (top-down): drill centroid + axis. Seeds the full top-down detect."""
        ym = self._mask(rgb, self.YELLOW_LO, self.YELLOW_HI)
        P = self._cloud(ym, depth, cam_pose)
        P = P[P[:, 2] > self.table_z + 0.01]
        if len(P) < 50:
            raise RuntimeError("drill not found")
        c = P[:, :2].mean(0)
        top = P[P[:, 2] > np.percentile(P[:, 2], 99.5) - 0.035]
        _, _, Vt = np.linalg.svd(top[:, :2] - top[:, :2].mean(0), full_matrices=False)
        axis = np.array([Vt[0, 0], Vt[0, 1]])
        S = self._cloud(self._mask(rgb, self.SILVER_LO, self.SILVER_HI, 0), depth, cam_pose)
        if len(S):
            rel = S[:, :2] - c
            S = S[(np.abs(rel @ np.array([-axis[1], axis[0]])) < 0.02) & (np.linalg.norm(rel, axis=1) < 0.3)
                  & (S[:, 2] > 0.1)]
            if len(S) >= 5 and np.median((S[:, :2] - c) @ axis) < 0:
                axis = -axis
        return c, axis

    def detect(self, rgb, depth, cam_pose, axis_hint=None):
        ym = self._mask(rgb, self.YELLOW_LO, self.YELLOW_HI)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(ym)
        if n < 2:
            raise RuntimeError("drill not found (no yellow blob)")
        big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        # keep every sizeable yellow blob near the biggest one (shadows can split the drill)
        x, y, w, h, _ = stats[big]; pad = int(0.5 * max(w, h))
        keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] > 30
                and stats[k, 0] + stats[k, 2] > x - pad and stats[k, 0] < x + w + pad
                and stats[k, 1] + stats[k, 3] > y - pad and stats[k, 1] < y + h + pad]
        ym = np.isin(lab, keep).astype(np.uint8) * 255
        P = self._cloud(ym, depth, cam_pose)
        P = P[P[:, 2] > self.table_z + 0.01]
        if len(P) < 50:
            raise RuntimeError(f"too few drill points ({len(P)})")

        # ---- housing (the top of the yellow cloud) -> drill axis
        z_top = np.percentile(P[:, 2], 99.5)
        H = P[P[:, 2] > z_top - 0.035]
        c = H[:, :2].mean(0)
        _, _, Vt = np.linalg.svd(H[:, :2] - c, full_matrices=False)
        axis = np.array([Vt[0, 0], Vt[0, 1], 0.0])

        # ---- silver bit fixes the axis sign and gives the tip
        sm = self._mask(rgb, self.SILVER_LO, self.SILVER_HI, open_k=0)
        # only accept silver pixels next to the drill blob (robot links are white too)
        x, y, w, h, _ = stats[big]
        roi = np.zeros_like(sm); pad = int(0.6 * max(w, h))
        roi[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad] = 255
        S = self._cloud(cv2.bitwise_and(sm, roi), depth, cam_pose)
        if len(S):
            rel = S[:, :2] - c
            lat = np.abs(rel @ np.array([-axis[1], axis[0]]))
            S = S[(lat < 0.02) & (np.abs(S[:, 2] - (z_top - 0.03)) < 0.03)
                  & (np.linalg.norm(rel, axis=1) < 0.30)]
        if len(S) >= 5:
            if np.median((S[:, :2] - c) @ axis[:2]) < 0:
                axis = -axis
        elif axis_hint is not None and np.dot(axis[:2], axis_hint[:2]) < 0:
            axis = -axis

        # ---- housing: circle fit in the plane perpendicular to the drill axis
        nrm = np.array([-axis[1], axis[0]])                 # horizontal, perpendicular to axis
        Hs = P[P[:, 2] > z_top - 0.052]                     # upper arc only (stay above the grip)
        Hs = Hs[np.abs((Hs[:, :2] - c) @ axis[:2]) < 0.05]  # middle of the housing only
        (l0, z_axis), r_house = fit_circle(np.column_stack([(Hs[:, :2] - c) @ nrm, Hs[:, 2]]), r_prior=0.03)
        r_house = float(np.clip(r_house, 0.02, 0.045))
        c = c + l0 * nrm                                    # centre the axis laterally
        underside = z_axis - r_house

        # ---- grip: yellow points below the housing and above the battery -> circle fit (top view)
        G = P[(P[:, 2] < underside - 0.008) & (P[:, 2] > self.table_z + 0.07)]
        self.dbg = dict(z_top=z_top, z_axis=z_axis, r_house=r_house, l0=l0, underside=underside,
                        nG_before=len(G), axis=axis.copy(), c=c.copy())
        G = G[np.abs((G[:, :2] - c) @ nrm) < 0.05]
        if len(G) < 20:
            raise RuntimeError(f"grip not visible ({len(G)} pts) - try a lower viewing angle")
        grip_xy, r_grip = fit_circle(G[:, :2], r_prior=0.0175, r_bounds=(0.010, 0.030))
        r_grip = float(r_grip)
        # Handle is coaxial with the housing in XY — drop lateral circle-fit bias
        # but keep the along-axis grip (the handle circle is a better origin than
        # the visible housing cloud, which is biased toward the chuck).
        ax2 = axis[:2] / (np.linalg.norm(axis[:2]) + 1e-12)
        along_g = float((grip_xy - c) @ ax2)
        grip_xy = c + along_g * ax2
        bit_tip = np.array([*(c + self.housing_to_tip * ax2), z_axis])

        est = DrillEstimate(grip_xy=grip_xy, axis=axis, housing_underside_z=float(underside),
                            housing_axis_z=float(z_axis), bit_tip=bit_tip, grip_radius=r_grip,
                            n_points=dict(drill=len(P), housing=len(H), grip=len(G), bit=len(S)))
        self.last_debug = dict(yellow=ym, silver=sm)
        return est

    def observe_held_tip(self, rgb, depth, cam_pose, tcp, axis, held_along, prior=None):
        """Live tip while held: housing centre + CAD reach along the bit axis.

        `held_along` here is `housing_to_tip` (not TCP→tip). Yellow depth is
        trusted; the thin silver bit often is not.
        """
        from task.tip import tip_from_housing_axis

        ym = self._mask(rgb, self.YELLOW_LO, self.YELLOW_HI)
        P = self._cloud(ym, depth, cam_pose)
        return tip_from_housing_axis(
            P, tcp, axis, housing_to_tip=held_along, prior=prior)

    def draw(self, rgb, est: DrillEstimate, cam_pose):
        """Overlay the estimate on the image (for the demo video)."""
        img = rgb.copy()
        if self.last_debug is not None:
            cnts, _ = cv2.findContours(self.last_debug["yellow"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, (0, 255, 0), 1)
        g0 = np.array([*est.grip_xy, 0.07]); g1 = np.array([*est.grip_xy, est.housing_underside_z])
        a0 = np.array([*est.grip_xy, est.housing_axis_z])
        pts, z = self.cam.project(np.stack([g0, g1, a0, est.bit_tip]), cam_pose)
        if np.all(z > 0):
            p = pts.astype(int)
            cv2.line(img, tuple(p[0]), tuple(p[1]), (255, 0, 255), 2)
            cv2.arrowedLine(img, tuple(p[2]), tuple(p[3]), (0, 128, 255), 2, tipLength=0.15)
            cv2.circle(img, tuple(p[3]), 4, (255, 0, 0), -1)
        cv2.putText(img, f"drill yaw {np.degrees(est.yaw):+.1f} deg", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img


@dataclass
class BoardEstimate:
    """Workpiece face from wrist RGB-D. `normal` points into the board (same as BOARD_N)."""
    point: np.ndarray          # point on the face (near the cloud centroid)
    normal: np.ndarray         # unit, into the board
    hole: np.ndarray           # target pierce point on the face
    n_points: int = 0
    inlier_ratio: float = 0.0
    target_source: str = "offset"   # "marker" | "aruco" | "offset"
    target_px: np.ndarray | None = None


class BoardDetector:
    # Wood-like HSV: sim plywood (rgba ~ 0.80, 0.64, 0.42) plus darker / less-saturated
    # real boards. Search-radius around the calib prior keeps the wider gate honest.
    WOOD_LO, WOOD_HI = (5, 20, 35), (40, 220, 255)
    # Dark target disk on a light pad (sim target_dot / printed marker).
    MARKER_LO, MARKER_HI = (0, 0, 0), (180, 80, 70)
    WHITE_LO, WHITE_HI = (0, 0, 180), (180, 50, 255)
    MARKER_RADIUS_M = 0.008
    MIN_CIRCULARITY = 0.65
    MIN_FILL = 0.55

    def __init__(self, cam: D435i, hole_offset_yz=(0.02, -0.01),
                 target_mode="marker", aruco_id=0):
        self.cam = cam
        self.hole_offset_yz = np.asarray(hole_offset_yz, float)
        self.target_mode = str(target_mode).lower()
        self.aruco_id = int(aruco_id)
        self.last_mask = None
        self.last_marker_mask = None

    def detect(self, rgb, depth, cam_pose, normal_hint=None, min_pts=400,
               search_center=None, search_radius=0.40, require_target=True):
        P, mk = self._board_cloud(
            rgb, depth, cam_pose, min_pts, search_center, search_radius)
        self.last_mask = mk

        center, normal, ratio = self._ransac_plane(P, iters=80, thresh=0.008)
        # convention: normal points into the board (away from the camera / toward +X)
        cam_p, _ = cam_pose
        if normal @ (center - cam_p) < 0:
            normal = -normal
        if normal_hint is not None and np.dot(normal, normal_hint) < 0:
            normal = -normal
        normal = normal / np.linalg.norm(normal)

        try:
            hole, source, px = self._resolve_hole(rgb, depth, cam_pose, center, normal)
        except RuntimeError:
            if require_target:
                raise
            hole, source, px = self._offset_hole(center, normal), "offset", None
        return BoardEstimate(point=center, normal=normal, hole=hole,
                             n_points=len(P), inlier_ratio=float(ratio),
                             target_source=source, target_px=px)

    def locate_target(self, rgb, depth, cam_pose, center, normal):
        """Find the pierce mark on an already-known board plane. Raises if missing."""
        return self._resolve_hole(rgb, depth, cam_pose, center, normal)

    def _board_cloud(self, rgb, depth, cam_pose, min_pts, search_center, search_radius):
        """Wood HSV first; if the stain/paint misses, RANSAC the depth in the search volume."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mk = cv2.inRange(hsv, np.array(self.WOOD_LO), np.array(self.WOOD_HI))
        mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        vs, us = np.nonzero((mk > 0) & (depth > 0))
        if len(us) >= min_pts:
            P = self.cam.deproject(us.astype(float), vs.astype(float), depth[vs, us], cam_pose)
            P = self._near_board_prior(P, search_center, search_radius)
            if len(P) >= min_pts:
                return P, mk
        vs, us = np.nonzero(depth > 0)
        if len(us) < min_pts:
            raise RuntimeError(f"board not found ({int((mk > 0).sum())} wood px, "
                               f"{len(us)} depth px)")
        P = self.cam.deproject(us.astype(float), vs.astype(float), depth[vs, us], cam_pose)
        P = self._near_board_prior(P, search_center, search_radius)
        if len(P) < min_pts:
            raise RuntimeError(f"board cloud too small after workspace filter ({len(P)})")
        return P, mk

    @staticmethod
    def _near_board_prior(P, search_center, search_radius):
        """Keep cloud points near the calibrated board pose, not a hardcoded sim box."""
        if search_center is not None:
            c = np.asarray(search_center, float)
            return P[np.linalg.norm(P - c, axis=1) < float(search_radius)]
        return P[(P[:, 0] > 0.35) & (P[:, 0] < 0.90) & (P[:, 2] > 0.05) & (P[:, 2] < 0.70)]

    def _offset_hole(self, center, normal):
        n, y, z = board_axes(normal)
        return (np.asarray(center, float)
                + float(self.hole_offset_yz[0]) * y
                + float(self.hole_offset_yz[1]) * z)

    def _project_to_plane(self, p, center, normal):
        p = np.asarray(p, float)
        return p - normal * float((p - center) @ normal)

    def _resolve_hole(self, rgb, depth, cam_pose, center, normal):
        mode = self.target_mode
        if mode == "offset":
            return self._offset_hole(center, normal), "offset", None
        if mode == "aruco":
            hit = self._detect_aruco(rgb, depth, cam_pose, center, normal)
            if hit is not None:
                return hit
            raise RuntimeError("ArUco board target not found")
        if mode == "marker":
            hit = self._detect_contrast_marker(rgb, depth, cam_pose, center, normal)
            if hit is not None:
                return hit
            raise RuntimeError("contrast board target not found")
        raise ValueError(f"unknown board target mode {mode!r}")

    def _detect_contrast_marker(self, rgb, depth, cam_pose, center, normal):
        """Black disk on a white pad. RGB circle + ray-plane; no centroid-offset guess."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        white = cv2.inRange(hsv, np.array(self.WHITE_LO), np.array(self.WHITE_HI))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cam_p, _ = cam_pose
        dist = abs(float((np.asarray(center) - cam_p) @ (
            np.asarray(normal) / (np.linalg.norm(normal) + 1e-12))))
        r_px_exp = float(self.cam.fx) * self.MARKER_RADIUS_M / max(dist, 0.08)
        k = max(9, int(2.2 * r_px_exp) | 1)
        pad = cv2.dilate(white, np.ones((k, k), np.uint8))
        dark = cv2.inRange(hsv, np.array(self.MARKER_LO), np.array(self.MARKER_HI))
        dark = cv2.bitwise_and(dark, pad)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        self.last_marker_mask = dark
        best = None
        cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            circ, area = contour_circularity(cnt)
            if circ < self.MIN_CIRCULARITY or area < 12:
                continue
            (u, v), r = cv2.minEnclosingCircle(cnt)
            if r < 2.5 or r > 90:
                continue
            fill = area / (np.pi * r * r + 1e-9)
            if fill < self.MIN_FILL:
                continue
            if not (0.35 * r_px_exp <= r <= 2.8 * r_px_exp):
                continue
            if not self._on_white_pad(white, u, v, r):
                continue
            try:
                hole = self.cam.ray_plane(u, v, center, normal, cam_pose)
            except RuntimeError:
                continue
            score = circ * fill * min(area, 800.0)
            if best is None or score > best[0]:
                best = (score, hole, np.array([u, v], float))
        hough = self._hough_marker(rgb, white, center, normal, cam_pose, r_px_exp)
        if hough is not None and (best is None or hough[0] > best[0]):
            best = hough
        if best is None:
            return None
        return best[1], "marker", best[2]

    @staticmethod
    def _on_white_pad(white, u, v, r):
        """True if a ring around the disk is mostly white (pad, not a random dark blob)."""
        h, w = white.shape
        rr = int(max(r + 5.0, r * 1.5))
        y0, y1 = max(0, int(v - rr)), min(h, int(v + rr) + 1)
        x0, x1 = max(0, int(u - rr)), min(w, int(u + rr) + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        d2 = (xx - u) ** 2 + (yy - v) ** 2
        ring = (d2 >= (0.85 * r) ** 2) & (d2 <= (1.7 * r) ** 2)
        if int(ring.sum()) < 8:
            return False
        return float(white[y0:y1, x0:x1][ring].mean()) > 127.0

    def _hough_marker(self, rgb, white, center, normal, cam_pose, r_px_exp):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
        rmin = max(3, int(0.4 * r_px_exp))
        rmax = max(rmin + 2, int(2.6 * r_px_exp))
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(8, int(r_px_exp)),
            param1=80, param2=12, minRadius=rmin, maxRadius=rmax)
        if circles is None:
            return None
        best = None
        for c in circles[0]:
            u, v, r = float(c[0]), float(c[1]), float(c[2])
            if not self._on_white_pad(white, u, v, r):
                continue
            try:
                hole = self.cam.ray_plane(u, v, center, normal, cam_pose)
            except RuntimeError:
                continue
            score = 0.85 * min(r, 40.0)
            if best is None or score > best[0]:
                best = (score, hole, np.array([u, v]))
        return best

    def _detect_aruco(self, rgb, depth, cam_pose, center, normal):
        if not hasattr(cv2, "aruco"):
            return None
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(rgb)
        if ids is None:
            return None
        ids = ids.reshape(-1)
        for i, mid in enumerate(ids):
            if int(mid) != self.aruco_id:
                continue
            c = corners[i][0]
            u, v = float(c[:, 0].mean()), float(c[:, 1].mean())
            # Close range: D435i depth is invalid below ~105 mm — use RGB ray-plane.
            try:
                hole = self.cam.ray_plane(u, v, center, normal, cam_pose)
            except RuntimeError:
                ui, vi = int(round(u)), int(round(v))
                z = float(depth[vi, ui]) if 0 <= vi < depth.shape[0] and 0 <= ui < depth.shape[1] and depth[vi, ui] > 0 else 0.0
                if z <= 0:
                    return None
                P = self.cam.deproject(np.array([u]), np.array([v]), np.array([z]), cam_pose)[0]
                hole = self._project_to_plane(P, center, normal)
            return hole, "aruco", np.array([u, v], float)
        return None

    @staticmethod
    def _ransac_plane(P, iters=80, thresh=0.008):
        rng = np.random.default_rng(0)
        best_inl = None
        n = len(P)
        for _ in range(iters):
            i = rng.choice(n, 3, replace=False)
            p0, p1, p2 = P[i]
            nrm = np.cross(p1 - p0, p2 - p0)
            ln = np.linalg.norm(nrm)
            if ln < 1e-9:
                continue
            nrm = nrm / ln
            d = np.abs((P - p0) @ nrm)
            inl = d < thresh
            if best_inl is None or inl.sum() > best_inl.sum():
                best_inl = inl
        if best_inl is None or best_inl.sum() < 50:
            raise RuntimeError("board plane fit failed")
        Q = P[best_inl]
        c = Q.mean(0)
        _, _, Vt = np.linalg.svd(Q - c, full_matrices=False)
        return c, Vt[2], best_inl.mean()

    def draw(self, rgb, est: BoardEstimate, cam_pose):
        img = rgb.copy()
        if self.last_mask is not None:
            tint = img.copy(); tint[self.last_mask > 0] = (40, 180, 255)
            img = cv2.addWeighted(img, 0.65, tint, 0.35, 0)
        if self.last_marker_mask is not None:
            tint = img.copy(); tint[self.last_marker_mask > 0] = (0, 255, 0)
            img = cv2.addWeighted(img, 0.75, tint, 0.25, 0)
        pts, z = self.cam.project(np.stack([est.hole, est.hole + 0.08 * est.normal]), cam_pose)
        if np.all(z > 0):
            p = pts.astype(int)
            cv2.circle(img, tuple(p[0]), 6, (0, 255, 0), -1)
            cv2.arrowedLine(img, tuple(p[0]), tuple(p[1]), (0, 255, 255), 2, tipLength=0.2)
        if est.target_px is not None:
            cv2.drawMarker(img, tuple(np.round(est.target_px).astype(int)), (255, 0, 255),
                           markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
        cv2.putText(img, f"board n=({est.normal[0]:+.2f},{est.normal[1]:+.2f},{est.normal[2]:+.2f}) "
                    f"hole={est.target_source}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return img
