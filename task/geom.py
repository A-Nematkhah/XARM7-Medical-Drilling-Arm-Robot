"""Geometry helpers for bit aiming (independent of MuJoCo)."""
import numpy as np

from .constants import UP


def mat2quat(R):
    """Rotation matrix → quaternion wxyz (same layout as MuJoCo)."""
    R = np.asarray(R, float).reshape(3, 3)
    t = float(np.trace(R))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def rodrigues(v, axis, deg):
    a = np.deg2rad(deg)
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return (np.eye(3) + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)) @ np.asarray(v, float)


def bit_direction(tilt_deg, azim_deg, board_n, up=UP):
    """Unit bit axis in world: 0/0 = board normal; +tilt=up, +azim=+Y."""
    n = np.asarray(board_n, float); n = n / np.linalg.norm(n)
    u = np.asarray(up, float); u = u / np.linalg.norm(u)
    d = rodrigues(n, u, azim_deg)
    tilt_axis = np.cross(d, u)
    if np.linalg.norm(tilt_axis) < 1e-8:
        raise ValueError("tilt undefined when bit is parallel to up")
    d = rodrigues(d, tilt_axis, tilt_deg)
    return d / np.linalg.norm(d)


def aim_rotation(d, board_n, up=UP):
    """Palm along the bit; fingers parallel to the board so they cannot stab into it."""
    d = np.asarray(d, float); d = d / np.linalg.norm(d)
    n = np.asarray(board_n, float); n = n / np.linalg.norm(n)
    u = np.asarray(up, float)
    z = np.cross(n, d)
    if np.linalg.norm(z) < 1e-6:
        y = u - d * np.dot(u, d)
        y = y / np.linalg.norm(y)
        return np.column_stack([d, y, np.cross(d, y)])
    z = z / np.linalg.norm(z)
    if np.dot(np.cross(z, d), u) < 0:
        z = -z
    y = np.cross(z, d); y = y / np.linalg.norm(y)
    return np.column_stack([d, y, z])


def quat_align_z(v):
    """Quaternion (wxyz) mapping local +Z to unit vector v."""
    z = np.asarray(v, float); z = z / np.linalg.norm(z)
    tmp = UP if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(tmp, z); x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return mat2quat(np.column_stack([x, y, z]))


def ang_err_deg(a, b):
    a = np.asarray(a, float); a = a / (np.linalg.norm(a) + 1e-12)
    b = np.asarray(b, float); b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))


def planar_err_mm(hit, target, normal, up=UP):
    n = np.asarray(normal, float); n = n / (np.linalg.norm(n) + 1e-12)
    d = np.asarray(hit, float) - np.asarray(target, float)
    lat = d - n * float(d @ n)
    u = np.asarray(up, float)
    y = np.cross(n, u)
    if np.linalg.norm(y) < 1e-8:
        y = np.cross(n, np.array([1.0, 0.0, 0.0]))
    y = y / (np.linalg.norm(y) + 1e-12)
    z = np.cross(y, n); z = z / (np.linalg.norm(z) + 1e-12)
    return float(np.linalg.norm(lat) * 1000.0), float(lat @ y * 1000.0), float(lat @ z * 1000.0)
