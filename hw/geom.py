"""TCP / xArm pose helpers (no MuJoCo). Metres + radians in, mm + deg for the SDK."""
from __future__ import annotations

import numpy as np


def rpy_deg_to_R(rpy_deg):
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll); matches xArm / calib convention."""
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy_deg, float).reshape(3))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def R_to_rpy_deg(R):
    """Inverse of `rpy_deg_to_R` (yaw-pitch-roll ZYX)."""
    R = np.asarray(R, float).reshape(3, 3)
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    cp = float(np.cos(pitch))
    if abs(cp) < 1e-8:
        roll = 0.0
        yaw = float(np.arctan2(-R[0, 1], R[1, 1]))
    else:
        roll = float(np.arctan2(R[2, 1] / cp, R[2, 2] / cp))
        yaw = float(np.arctan2(R[1, 0] / cp, R[0, 0] / cp))
    return np.degrees([roll, pitch, yaw])


def pose_m_R_to_xarm_mm_rpy(p_m, R):
    """World TCP (m, 3x3) -> xArm set_position args (mm + RPY deg)."""
    p = np.asarray(p_m, float).reshape(3)
    rpy = R_to_rpy_deg(R)
    return [float(p[0] * 1000.0), float(p[1] * 1000.0), float(p[2] * 1000.0),
            float(rpy[0]), float(rpy[1]), float(rpy[2])]


def xarm_mm_rpy_to_pose_m_R(pose6):
    """xArm get_position [x,y,z,roll,pitch,yaw] -> (p_m, R)."""
    x, y, z, roll, pitch, yaw = [float(v) for v in pose6[:6]]
    p = np.array([x, y, z], float) * 0.001
    R = rpy_deg_to_R([roll, pitch, yaw])
    return p, R
