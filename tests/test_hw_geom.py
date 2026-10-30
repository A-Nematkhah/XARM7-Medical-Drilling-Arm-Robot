import numpy as np

from hw.geom import R_to_rpy_deg, pose_m_R_to_xarm_mm_rpy, rpy_deg_to_R, xarm_mm_rpy_to_pose_m_R


def test_rpy_roundtrip():
    for rpy in ([0, 0, 0], [0, -90, 0], [10, -20, 35], [-45, 15, 90]):
        R = rpy_deg_to_R(rpy)
        back = R_to_rpy_deg(R)
        assert np.allclose(rpy_deg_to_R(back), R, atol=1e-9)


def test_xarm_pose_roundtrip():
    p = np.array([0.4, -0.1, 0.25])
    R = rpy_deg_to_R([10.0, -30.0, 45.0])
    pose6 = pose_m_R_to_xarm_mm_rpy(p, R)
    p2, R2 = xarm_mm_rpy_to_pose_m_R(pose6)
    assert np.allclose(p, p2, atol=1e-9)
    assert np.allclose(R, R2, atol=1e-9)
