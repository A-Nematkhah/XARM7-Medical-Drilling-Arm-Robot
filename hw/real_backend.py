"""Real-robot backend stubs (xArm SDK / RH56 / D435i).

No MuJoCo imports. Live Cartesian/joint motion must go to the controller
(`set_position` / `set_servo_angle`); do not call Python `servo_step` or
`mj_jacSite` here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hw.defaults import REST_Q


class RealBackendNotReady(NotImplementedError):
    """Raised until a real SDK is wired behind the shared interfaces."""


def _need_sdk(what: str):
    raise RealBackendNotReady(
        f"{what}: real SDK not wired. Use --backend sim until xArm/RH56/D435i "
        "are connected. Live motion must use the controller API, not Python IK."
    )


class RealArm:
    """xArm7 controller driver. Matches `hw.interfaces.ArmDriver`."""

    def __init__(self, arm_ip: str = "192.168.1.xxx"):
        self.arm_ip = arm_ip
        self.max_track_err = 0.0
        self.callbacks: list = []
        self.rest_q = REST_Q.copy()
        self.qadr = np.arange(7)  # logical joint indices for planning helpers
        self._q = REST_Q.copy()
        self._p = np.zeros(3)
        self._R = np.eye(3)

    def get_q(self):
        """Measured joint angles from the controller (stub: last commanded)."""
        return self._q.copy()

    def site_pose(self):
        """TCP pose from the controller / FK (stub until SDK)."""
        _need_sdk("RealArm.site_pose")

    def wait(self, t: float) -> None:
        _need_sdk("RealArm.wait")

    def move_joints(self, q_goal, T: float = 2.0) -> None:
        # Wire: arm.set_servo_angle(angle=q_goal, speed=..., wait=True)
        _need_sdk("RealArm.move_joints")

    def move_to_pose(self, p_goal, R_goal, T: float = 2.0, settle: float = 0.3, gain: float = 0.6):
        # Wire: arm.set_position(...); never Robot.servo_step / mj_jacSite
        _need_sdk("RealArm.move_to_pose")

    def limit_margin(self, q) -> float:
        # Replace with measured joint limits from the controller manual / SDK.
        lo = np.array([-6.28, -2.06, -6.28, -0.19, -6.28, -1.69, -6.28])
        hi = np.array([6.28, 2.09, 6.28, 3.93, 6.28, 3.14, 6.28])
        q = np.asarray(q, float)
        lim = (hi - lo) < 12.0
        return float(np.min(np.minimum(q - lo, hi - q)[lim]))

    def plan_path_ik(self, waypoints, R, min_margin: float = 0.2, max_jump: float = 1.0):
        # Wire: offline xArm get_inverse_kinematics along waypoints (not live DLS).
        _need_sdk("RealArm.plan_path_ik")

    def move_joint_path(self, chain, T_first: float = 3.0, T_step: float = 0.8):
        _need_sdk("RealArm.move_joint_path")

    def ik_solutions(self, p_goal, R_goal, n_random: int = 16):
        _need_sdk("RealArm.ik_solutions")

    def goto_pose_joint_space(self, p_goal, R_goal, T: float = 2.5, seed=None):
        # Wire: IK on controller, then move_joints — still no Python servo_step.
        _need_sdk("RealArm.goto_pose_joint_space")


class RealHand:
    """RH56 serial driver stub. Matches `hw.interfaces.HandDriver`."""

    OPEN = [1000, 1000, 1000, 1000, 1000, 1000]
    PRE_GRASP = [1000, 1000, 1000, 1000, 1000, 250]

    def __init__(self, port: str | None = None, force_scale: float = 1.0):
        self.port = port
        self.force_scale = float(force_scale)
        self.target = list(self.OPEN)
        self.cmd = list(self.OPEN)

    def set_angles(self, angles, speed=None) -> None:
        _need_sdk("RealHand.set_angles")

    def get_angles(self):
        _need_sdk("RealHand.get_angles")

    def get_forces(self):
        _need_sdk("RealHand.get_forces")


class RealCamera:
    """pyrealsense2 wrist D435i stub. Matches `hw.interfaces.WristCamera`."""

    def __init__(self, hand_eye=None, tcp_pose_fn=None, min_z=0.105):
        self.hand_eye = hand_eye
        self.tcp_pose_fn = tcp_pose_fn
        self.min_z = float(min_z)
        self.fx = self.fy = 469.0
        self.cx, self.cy = 319.5, 179.5

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def flush(self):
        pass

    def settle_capture(self):
        return self.capture()

    def capture(self):
        _need_sdk("RealCamera.capture")

    def pose(self):
        if self.hand_eye is not None and self.tcp_pose_fn is not None:
            from calibration import camera_pose_from_tcp
            p_ee, R_ee = self.tcp_pose_fn()
            return camera_pose_from_tcp(p_ee, R_ee, self.hand_eye)
        _need_sdk("RealCamera.pose")

    def pixel_ray(self, u, v, cam_pose=None):
        p, R = self.pose() if cam_pose is None else cam_pose
        dir_cam = np.array([(u - self.cx) / self.fx, -(v - self.cy) / self.fy, -1.0])
        d = R @ dir_cam
        return p, d / (np.linalg.norm(d) + 1e-12)

    def ray_plane(self, u, v, center, normal, cam_pose=None):
        from calibration import intersect_ray_plane
        return intersect_ray_plane(*self.pixel_ray(u, v, cam_pose), center, normal)

    def deproject(self, us, vs, zs, cam_pose=None):
        _need_sdk("RealCamera.deproject")

    def project(self, P, cam_pose=None):
        _need_sdk("RealCamera.project")


@dataclass
class RealBackend:
    arm: RealArm
    hand: RealHand
    camera: RealCamera
    source: str = "real"


def make_real_backend(*, arm_ip: str = "192.168.1.xxx", hand_port: str | None = None,
                      hand_eye=None, force_scale: float = 1.0, min_z: float = 0.105) -> RealBackend:
    """Return a real backend whose motion/capture methods raise until the SDK is wired."""
    arm = RealArm(arm_ip=arm_ip)
    hand = RealHand(port=hand_port, force_scale=force_scale)
    cam = RealCamera(hand_eye=hand_eye, tcp_pose_fn=arm.site_pose, min_z=min_z)
    return RealBackend(arm=arm, hand=hand, camera=cam)
