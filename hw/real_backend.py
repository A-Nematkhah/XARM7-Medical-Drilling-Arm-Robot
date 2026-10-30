"""Real-robot backends (xArm SDK / RH56 serial / D435i via pyrealsense2).

No MuJoCo imports. Live Cartesian/joint motion goes to the controller
(`set_position` / `set_servo_angle`); never Python `servo_step` / `mj_jacSite`.

SDKs are optional: without them (or without a live robot) methods raise
`RealBackendNotReady`. With packages installed and `hardware` filled in
`config/calib.yaml`, the same `ArmDriver` / `HandDriver` / `WristCamera`
contracts talk to hardware.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from hw.defaults import REST_Q, XARM7_HI, XARM7_LO
from hw.geom import pose_m_R_to_xarm_mm_rpy, xarm_mm_rpy_to_pose_m_R
from hw.ik_select import limit_margin as _limit_margin, pick_best_chain, rank_qs


class RealBackendNotReady(NotImplementedError):
    """Raised when an SDK is missing, disconnected, or a call is not wired."""


def _need_sdk(what: str, detail: str = ""):
    extra = f" ({detail})" if detail else ""
    raise RealBackendNotReady(
        f"{what}: real SDK not ready{extra}. Install requirements-hw.txt, set "
        "hardware.* in config/calib.yaml, and use --backend real on the robot PC. "
        "Live motion must use the controller API, not Python IK."
    )


def _check_xarm_code(code, what: str):
    if code is None:
        return
    if isinstance(code, (tuple, list)):
        code = code[0]
    if int(code) != 0:
        raise RealBackendNotReady(f"{what}: xArm error code {code}")


def cart_speed_mm_s(dist_m, T, tcp_speed_mm_s, vmax_m_s=None) -> float:
    """Cartesian speed sent to xArm `set_position`, in mm/s.

    Never raises a derated/gated cap back up to a floor (old code used 10 mm/s,
    which undid stall-derated feeds below that). 1 mm/s is only a SDK minimum.
    """
    cap = float(tcp_speed_mm_s)
    if vmax_m_s is not None:
        cap = min(cap, float(vmax_m_s) * 1000.0)
    need = (float(dist_m) * 1000.0) / max(float(T), 0.1)
    return min(cap, max(1.0, need))


def _try_import_xarm():
    try:
        from xarm.wrapper import XArmAPI  # type: ignore
        return XArmAPI
    except ImportError:
        return None


def _try_import_rs():
    try:
        import pyrealsense2 as rs  # type: ignore
        return rs
    except ImportError:
        return None


def _try_import_serial():
    try:
        import serial  # type: ignore
        return serial
    except ImportError:
        return None


class RealArm:
    """xArm7 controller driver. Matches `hw.interfaces.ArmDriver`."""

    def __init__(self, arm_ip: str = "192.168.1.xxx", *,
                 connect: bool = False, tcp_speed_mm_s: float = 80.0,
                 joint_speed_deg_s: float = 30.0):
        self.arm_ip = str(arm_ip)
        self.tcp_speed_mm_s = float(tcp_speed_mm_s)
        self.joint_speed_deg_s = float(joint_speed_deg_s)
        self.max_track_err = 0.0
        self.callbacks: list = []
        self.rest_q = REST_Q.copy()
        self.qadr = np.arange(7)
        self._q = REST_Q.copy()
        self._p = np.zeros(3)
        self._R = np.eye(3)
        self._arm = None
        self._XArmAPI = _try_import_xarm()
        if connect:
            self.connect()

    @property
    def connected(self) -> bool:
        return self._arm is not None

    def connect(self):
        if self._arm is not None:
            return self
        if self._XArmAPI is None:
            _need_sdk("RealArm.connect", "xarm.wrapper.XArmAPI missing")
        if "xxx" in self.arm_ip.lower() or not self.arm_ip.strip():
            _need_sdk("RealArm.connect", "set hardware.arm_ip in calib.yaml")
        arm = self._XArmAPI(self.arm_ip, is_radian=True)
        _check_xarm_code(arm.motion_enable(enable=True), "motion_enable")
        _check_xarm_code(arm.set_mode(0), "set_mode")
        _check_xarm_code(arm.set_state(0), "set_state")
        self._arm = arm
        self._refresh_state()
        return self

    def _ensure(self):
        if self._arm is None:
            self.connect()
        return self._arm

    def _refresh_state(self):
        arm = self._arm
        if arm is None:
            return
        code, angles = arm.get_servo_angle(is_radian=True)
        _check_xarm_code(code, "get_servo_angle")
        self._q = np.asarray(angles[:7], float)
        code, pose = arm.get_position(is_radian=False)
        _check_xarm_code(code, "get_position")
        self._p, self._R = xarm_mm_rpy_to_pose_m_R(pose)

    def get_q(self):
        if self._arm is not None:
            self._refresh_state()
        return self._q.copy()

    def site_pose(self):
        arm = self._ensure()
        code, pose = arm.get_position(is_radian=False)
        _check_xarm_code(code, "site_pose")
        self._p, self._R = xarm_mm_rpy_to_pose_m_R(pose)
        return self._p.copy(), self._R.copy()

    def wait(self, t: float) -> None:
        self._ensure()
        time.sleep(max(0.0, float(t)))
        for cb in self.callbacks:
            cb()

    def joint_speed_rad_s(self, q_goal, T: float = 2.0) -> float:
        """rad/s for `set_servo_angle(..., is_radian=True)`. Cap is `joint_speed_deg_s`."""
        dq = float(np.max(np.abs(np.asarray(q_goal, float).reshape(7) - self.get_q())))
        vmax = float(np.deg2rad(max(self.joint_speed_deg_s, 1.0)))
        return float(max(0.05, min(vmax, dq / max(float(T), 0.1))))

    def move_joints(self, q_goal, T: float = 2.0) -> None:
        arm = self._ensure()
        q = np.asarray(q_goal, float).reshape(7)
        speed = self.joint_speed_rad_s(q, T)
        code = arm.set_servo_angle(angle=q.tolist(), speed=speed, is_radian=True, wait=True)
        _check_xarm_code(code, "move_joints")
        self._q = q.copy()
        self._refresh_state()
        for cb in self.callbacks:
            cb()

    def move_to_pose(self, p_goal, R_goal, T: float = 2.0, settle: float = 0.3,
                     gain: float = 0.6, vmax=None):
        del settle, gain  # controller closes the loop; gain is sim-only
        arm = self._ensure()
        p_goal = np.asarray(p_goal, float).reshape(3)
        x, y, z, roll, pitch, yaw = pose_m_R_to_xarm_mm_rpy(p_goal, R_goal)
        dist = float(np.linalg.norm(p_goal - self._p))
        speed = cart_speed_mm_s(dist, T, self.tcp_speed_mm_s, vmax)
        code = arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
                                speed=speed, wait=True, is_radian=False)
        _check_xarm_code(code, "move_to_pose")
        self._refresh_state()
        err = p_goal - self._p
        self.max_track_err = float(np.linalg.norm(err))
        for cb in self.callbacks:
            cb()
        return np.concatenate([err, np.zeros(3)])

    def move_along(self, direction, distance_m, R_hold, T: float = 1.0, speed=None, settle: float = 0.05):
        """Linear TCP hop along `direction` (controller Cartesian move, not Python IK)."""
        del settle
        d = np.asarray(direction, float)
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-12:
            raise ValueError("move_along: direction has zero length")
        d = d / nrm
        p0, _ = self.site_pose()
        vmax = float(speed) if speed is not None else self.tcp_speed_mm_s / 1000.0
        dist = float(distance_m)
        T_use = max(float(T), abs(dist) / max(vmax, 1e-6))
        return self.move_to_pose(p0 + dist * d, R_hold, T=T_use, vmax=vmax)

    def limit_margin(self, q) -> float:
        return _limit_margin(q, XARM7_LO, XARM7_HI)

    def _ik_one(self, p_goal, R_goal, seed=None):
        arm = self._ensure()
        pose = pose_m_R_to_xarm_mm_rpy(p_goal, R_goal)
        code, angles = arm.get_inverse_kinematics(pose, input_is_radian=False, return_is_radian=True)
        if int(code) != 0:
            return None
        q = np.asarray(angles[:7], float)
        if seed is not None and np.max(np.abs(q - np.asarray(seed, float))) > 2.5:
            return None
        return q

    def plan_path_ik(self, waypoints, R, min_margin: float = 0.2, max_jump: float = 1.0):
        """Offline controller IK along waypoints (not live DLS / mj_jacSite)."""
        self._ensure()
        sols0 = self.ik_solutions(waypoints[0], R, n_random=8)
        if not sols0:
            raise RuntimeError("no continuous IK chain along the approach path")
        chains = []
        for q0 in sols0:
            chain, q, ok = [q0], q0, True
            for w in waypoints[1:]:
                qn = self._ik_one(w, R, seed=q)
                if qn is None or np.max(np.abs(qn - q)) > max_jump:
                    ok = False
                    break
                chain.append(qn)
                q = qn
            if ok:
                chains.append(chain)
        if not chains:
            raise RuntimeError("no continuous IK chain along the approach path")
        picked = pick_best_chain(
            chains, self.get_q(), rest=self.rest_q, lo=XARM7_LO, hi=XARM7_HI,
            min_margin=min_margin)
        if picked is None:
            best_m = max(min(self.limit_margin(c) for c in ch) for ch in chains)
            raise RuntimeError(f"best joint-limit margin only {best_m:.2f} rad")
        return picked

    def move_joint_path(self, chain, T_first: float = 3.0, T_step: float = 0.8):
        if not chain:
            return
        self.move_joints(chain[0], T=T_first)
        for q in chain[1:]:
            self.move_joints(q, T=T_step)

    def ik_solutions(self, p_goal, R_goal, n_random: int = 16):
        self._ensure()
        cur = self.get_q()
        found = []
        q = self._ik_one(p_goal, R_goal, seed=cur)
        if q is not None:
            found.append(q)
        # Controller IK is single-seed; perturb joint seeds via small TCP offsets.
        rng = np.random.default_rng(1)
        for _ in range(max(0, int(n_random))):
            jitter = rng.normal(0, 0.004, 3)
            qj = self._ik_one(np.asarray(p_goal, float) + jitter, R_goal, seed=cur)
            if qj is not None and all(np.linalg.norm(qj - s) > 0.1 for s in found):
                found.append(qj)
        return rank_qs(found, cur, rest=self.rest_q, lo=XARM7_LO, hi=XARM7_HI)

    def goto_pose_joint_space(self, p_goal, R_goal, T: float = 2.5, seed=None):
        self._ensure()
        q = self._ik_one(p_goal, R_goal, seed=seed if seed is not None else self.get_q())
        if q is None:
            raise RuntimeError("IK failed on controller")
        self.move_joints(q, T=T)
        return q


class RealHand:
    """RH56 Modbus-RTU driver (angle registers). Matches `hw.interfaces.HandDriver`."""

    OPEN = [1000, 1000, 1000, 1000, 1000, 1000]
    PRE_GRASP = [1000, 1000, 1000, 1000, 1000, 250]

    # Default Inspire RH56 register map (angle setpoints, 0=closed .. 1000=open).
    ANGLE_REG_START = 1486
    FORCE_REG_START = 1586

    def __init__(self, port: str | None = None, force_scale: float = 1.0,
                 *, baudrate: int = 115200, slave_id: int = 1, connect: bool = False):
        self.port = port
        self.baudrate = int(baudrate)
        self.slave_id = int(slave_id)
        self.force_scale = float(force_scale)
        self.target = list(self.OPEN)
        self.cmd = list(self.OPEN)
        self._ser = None
        self._serial_mod = _try_import_serial()
        if connect:
            self.connect()

    @property
    def connected(self) -> bool:
        return self._ser is not None and getattr(self._ser, "is_open", False)

    def connect(self):
        if self.connected:
            return self
        if self._serial_mod is None:
            _need_sdk("RealHand.connect", "pyserial missing")
        if not self.port:
            _need_sdk("RealHand.connect", "set hardware.hand_port in calib.yaml")
        self._ser = self._serial_mod.Serial(
            self.port, baudrate=self.baudrate, bytesize=8, parity="N",
            stopbits=1, timeout=0.2)
        return self

    def _ensure(self):
        if not self.connected:
            self.connect()
        return self._ser

    @staticmethod
    def _crc(data: bytes) -> bytes:
        crc = 0xFFFF
        for b in data:
            crc ^= b
            for _ in range(8):
                crc = (crc >> 1) ^ 0xA001 if (crc & 1) else (crc >> 1)
        return bytes((crc & 0xFF, (crc >> 8) & 0xFF))

    def _write_regs(self, start: int, values):
        ser = self._ensure()
        vals = [max(0, min(1000, int(v))) for v in values]
        n = len(vals)
        payload = bytearray([
            self.slave_id & 0xFF, 0x10,
            (start >> 8) & 0xFF, start & 0xFF,
            (n >> 8) & 0xFF, n & 0xFF,
            n * 2,
        ])
        for v in vals:
            payload.append((v >> 8) & 0xFF)
            payload.append(v & 0xFF)
        frame = bytes(payload) + self._crc(payload)
        ser.write(frame)
        ser.flush()
        time.sleep(0.03)
        _ = ser.read(8)

    def _read_regs(self, start: int, n: int) -> list[int]:
        ser = self._ensure()
        payload = bytes([
            self.slave_id & 0xFF, 0x03,
            (start >> 8) & 0xFF, start & 0xFF,
            (n >> 8) & 0xFF, n & 0xFF,
        ])
        frame = payload + self._crc(payload)
        ser.reset_input_buffer()
        ser.write(frame)
        ser.flush()
        raw = ser.read(5 + 2 * n)
        if len(raw) < 5 + 2 * n:
            _need_sdk("RealHand._read_regs", f"short Modbus reply ({len(raw)} B)")
        out = []
        for i in range(n):
            hi, lo = raw[3 + 2 * i], raw[4 + 2 * i]
            out.append((hi << 8) | lo)
        return out

    def set_angles(self, angles, speed=None) -> None:
        del speed
        vals = [int(a) for a in angles]
        if len(vals) != 6:
            raise ValueError("RH56 expects 6 finger angles")
        self.target = list(vals)
        self._write_regs(self.ANGLE_REG_START, vals)
        self.cmd = list(vals)

    def get_angles(self):
        try:
            self.cmd = self._read_regs(self.ANGLE_REG_START, 6)
        except RealBackendNotReady:
            if self._ser is None and self._serial_mod is None:
                _need_sdk("RealHand.get_angles", "pyserial missing")
            raise
        return list(self.cmd)

    def get_forces(self):
        raw = self._read_regs(self.FORCE_REG_START, 6)
        return np.asarray(raw, float) * self.force_scale


class RealCamera:
    """pyrealsense2 wrist D435i. Matches `hw.interfaces.WristCamera`."""

    def __init__(self, hand_eye=None, tcp_pose_fn=None, min_z=0.105,
                 *, serial: str | None = None, width: int = 640, height: int = 360,
                 fps: int = 30, connect: bool = False,
                 base_eye=None, base_pose_fn=None):
        self.hand_eye = hand_eye
        self.tcp_pose_fn = tcp_pose_fn
        self.base_eye = base_eye
        self.base_pose_fn = base_pose_fn
        self.min_z = float(min_z)
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.fx = self.fy = 469.0
        self.cx, self.cy = (width - 1) / 2.0, (height - 1) / 2.0
        self._rs = _try_import_rs()
        self._pipe = None
        self._align = None
        if connect:
            self.connect()

    @property
    def connected(self) -> bool:
        return self._pipe is not None

    def connect(self):
        if self._pipe is not None:
            return self
        if self._rs is None:
            _need_sdk("RealCamera.connect", "pyrealsense2 missing")
        rs = self._rs
        pipe = rs.pipeline()
        cfg = rs.config()
        if self.serial:
            cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        profile = pipe.start(cfg)
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.fx, self.fy = float(intr.fx), float(intr.fy)
        self.cx, self.cy = float(intr.ppx), float(intr.ppy)
        self._align = rs.align(rs.stream.color)
        self._pipe = pipe
        return self

    def _ensure(self):
        if self._pipe is None:
            self.connect()
        return self._pipe

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def flush(self):
        if self._pipe is None:
            return
        for _ in range(4):
            try:
                self._pipe.wait_for_frames(timeout_ms=100)
            except Exception:
                break

    def settle_capture(self):
        self.flush()
        return self.capture()

    def capture(self):
        pipe = self._ensure()
        frames = pipe.wait_for_frames(timeout_ms=5000)
        frames = self._align.process(frames)
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data()).astype(np.float32) * 0.001
        depth[depth < self.min_z] = 0.0
        # Task expects RGB; RealSense color stream here is BGR.
        rgb = color[:, :, ::-1].copy()
        return rgb, depth

    def pose(self):
        if self.hand_eye is not None and self.tcp_pose_fn is not None:
            from calibration import camera_pose_from_tcp
            p_ee, R_ee = self.tcp_pose_fn()
            return camera_pose_from_tcp(p_ee, R_ee, self.hand_eye)
        if self.base_eye is not None:
            from calibration import camera_pose_from_base
            if self.base_pose_fn is not None:
                p_b, R_b = self.base_pose_fn()
            else:
                p_b, R_b = np.zeros(3), np.eye(3)
            return camera_pose_from_base(p_b, R_b, self.base_eye)
        _need_sdk("RealCamera.pose", "hand_eye + tcp_pose_fn or base_eye required")

    def pixel_ray(self, u, v, cam_pose=None):
        p, R = self.pose() if cam_pose is None else cam_pose
        dir_cam = np.array([(u - self.cx) / self.fx, -(v - self.cy) / self.fy, -1.0])
        d = R @ dir_cam
        return p, d / (np.linalg.norm(d) + 1e-12)

    def ray_plane(self, u, v, center, normal, cam_pose=None):
        from calibration import intersect_ray_plane
        return intersect_ray_plane(*self.pixel_ray(u, v, cam_pose), center, normal)

    def deproject(self, us, vs, zs, cam_pose=None):
        p, R = self.pose() if cam_pose is None else cam_pose
        us = np.asarray(us, float).reshape(-1)
        vs = np.asarray(vs, float).reshape(-1)
        zs = np.asarray(zs, float).reshape(-1)
        x = (us - self.cx) / self.fx * zs
        y = -(vs - self.cy) / self.fy * zs
        pts_cam = np.column_stack([x, y, -zs])
        return (R @ pts_cam.T).T + p

    def project(self, P, cam_pose=None):
        """Match sim `D435i.project`: `(pts[N,2], z[N])`."""
        p, R = self.pose() if cam_pose is None else cam_pose
        P = np.asarray(P, float).reshape(-1, 3)
        Pc = (R.T @ (P - p).T).T
        z = -Pc[:, 2]
        u = self.cx + self.fx * Pc[:, 0] / np.maximum(z, 1e-9)
        v = self.cy - self.fy * Pc[:, 1] / np.maximum(z, 1e-9)
        return np.stack([u, v], axis=1), z

    def close(self):
        if self._pipe is not None:
            try:
                self._pipe.stop()
            finally:
                self._pipe = None


class RealLidar:
    """Wrist 360° LiDAR stub. Matches `hw.interfaces.WristLidarDriver`."""

    def __init__(self, lidar_eye=None, tcp_pose_fn=None):
        self.lidar_eye = lidar_eye
        self.tcp_pose_fn = tcp_pose_fn

    def pose(self):
        if self.lidar_eye is None:
            _need_sdk("RealLidar.pose", "lidar_eye missing")
        from calibration import camera_pose_from_tcp
        if self.tcp_pose_fn is not None:
            p_ee, R_ee = self.tcp_pose_fn()
        else:
            p_ee, R_ee = np.zeros(3), np.eye(3)
        return camera_pose_from_tcp(p_ee, R_ee, self.lidar_eye)

    def scan(self):
        _need_sdk("RealLidar.scan", "wrist 360° LiDAR driver not wired")

    def locate(self, scan=None):
        _need_sdk("RealLidar.locate", "wrist 360° LiDAR driver not wired")


@dataclass
class RealBackend:
    arm: RealArm
    hand: RealHand
    camera: RealCamera
    lidar: RealLidar | None = None
    source: str = "real"


def make_real_backend(*, arm_ip: str = "192.168.1.xxx", hand_port: str | None = None,
                      hand_eye=None, force_scale: float = 1.0, min_z: float = 0.105,
                      rs_serial: str | None = None, hand_baud: int = 115200,
                      hand_slave_id: int = 1, tcp_speed_mm_s: float = 80.0,
                      joint_speed_deg_s: float = 30.0, connect: bool = False,
                      lidar_eye=None, **_ignored) -> RealBackend:
    """Build real drivers. `connect=True` opens devices immediately; default is lazy."""
    if float(force_scale) >= 0.1:
        import warnings
        warnings.warn(
            f"hand.force_scale={force_scale} looks like sim N*m units; RH56 FORCE_ACT is "
            "0..1000. Set force_scale so get_forces()*scale is in N*m (e.g. 0.001).",
            stacklevel=2)
    arm = RealArm(arm_ip=arm_ip, connect=False, tcp_speed_mm_s=tcp_speed_mm_s,
                  joint_speed_deg_s=joint_speed_deg_s)
    hand = RealHand(port=hand_port, force_scale=force_scale, baudrate=hand_baud,
                    slave_id=hand_slave_id, connect=False)
    cam = RealCamera(hand_eye=hand_eye, tcp_pose_fn=arm.site_pose, min_z=min_z,
                     serial=rs_serial, connect=False)
    if connect:
        arm.connect()
        hand.connect()
        cam.connect()
    lidar = (RealLidar(lidar_eye=lidar_eye, tcp_pose_fn=arm.site_pose)
             if lidar_eye is not None else None)
    return RealBackend(arm=arm, hand=hand, camera=cam, lidar=lidar)
