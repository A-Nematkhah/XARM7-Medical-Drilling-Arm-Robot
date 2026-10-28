"""MuJoCo-backed drivers. Thin wrappers so the task can swap backends later."""
from __future__ import annotations

from dataclasses import dataclass

from perception import D435i
from robot import Robot


@dataclass
class SimBackend:
    arm: Robot
    hand: object
    camera: D435i


def make_sim_backend(model, data, *, noise=True, seed=0, site="grasp_site",
                     hand_eye=None, force_scale=1.0, camera_calib=None) -> SimBackend:
    arm = Robot(model, data, site=site)
    arm.hand.force_scale = float(force_scale)
    cam_kw = {}
    if camera_calib is not None and noise:
        cam_kw = dict(
            latency_s=getattr(camera_calib, "latency_s", 0.0),
            dropout=getattr(camera_calib, "dropout", 0.01),
            edge_noise_m=getattr(camera_calib, "edge_noise_m", 0.0),
        )
    # --no-noise / ideal depth: no delay line (deterministic eval).
    if camera_calib is not None:
        cam_kw["min_z"] = float(getattr(camera_calib, "min_z", 0.105))
    cam = D435i(model, data, noise=noise, seed=seed,
                hand_eye=hand_eye, tcp_pose_fn=arm.site_pose, **cam_kw)
    return SimBackend(arm=arm, hand=arm.hand, camera=cam)
