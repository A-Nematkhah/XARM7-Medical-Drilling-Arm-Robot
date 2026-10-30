import cv2
import numpy as np
import pytest

from perception import BoardDetector, contour_circularity, intersect_ray_plane
from task.phases.approach import _planar_miss


class _CloudCam:
    def deproject(self, us, vs, zs, cam_pose):
        # Map pixel u to X around the board prior.
        P = np.column_stack([0.64 + 0.001 * (us - us.mean()),
                             -0.34 + 0.001 * (vs - vs.mean()),
                             np.full(len(us), 0.34)])
        return P


def test_board_cloud_falls_back_to_depth_when_wood_hsv_misses():
    rgb = np.full((20, 20, 3), (20, 40, 180), np.uint8)  # blue, not wood
    depth = np.full((20, 20), 0.40, np.float32)
    det = BoardDetector(_CloudCam(), target_mode="offset")
    P, mk = det._board_cloud(rgb, depth, cam_pose=None, min_pts=50,
                             search_center=np.array([0.64, -0.34, 0.34]),
                             search_radius=0.40)
    assert len(P) >= 50
    assert int(mk.sum()) == 0  # wood mask empty; depth fallback used


def test_near_board_prior_filters_around_calib_pose():
    P = np.array([
        [0.64, -0.34, 0.34],
        [1.20, 0.50, 0.90],
    ])
    near = BoardDetector._near_board_prior(P, np.array([0.64, -0.34, 0.34]), 0.40)
    assert len(near) == 1
    assert np.allclose(near[0], P[0])


def test_detect_require_target_false_falls_back_to_offset(monkeypatch):
    det = BoardDetector(cam=None, hole_offset_yz=(0.02, -0.01), target_mode="marker")
    rgb = np.zeros((8, 8, 3), np.uint8)
    depth = np.ones((8, 8), np.float32)
    pose = (np.array([0.2, 0.0, 0.3]), np.eye(3))
    P = np.array([[0.64, y, z] for y in np.linspace(-0.05, 0.05, 12)
                  for z in np.linspace(0.30, 0.38, 12)])
    monkeypatch.setattr(det, "_board_cloud", lambda *a, **k: (P, np.zeros((8, 8), np.uint8)))
    monkeypatch.setattr(det, "_resolve_hole",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no mark")))
    with pytest.raises(RuntimeError, match="no mark"):
        det.detect(rgb, depth, pose, require_target=True)
    est = det.detect(rgb, depth, pose, require_target=False)
    assert est.target_source == "offset"
    assert abs((est.hole - est.point) @ est.normal) < 1e-6


def test_offset_hole_lies_in_plane():
    det = BoardDetector(cam=None, hole_offset_yz=(0.02, -0.01), target_mode="offset")
    center = np.array([0.628, -0.34, 0.34])
    normal = np.array([1.0, 0.0, 0.0])
    hole = det._offset_hole(center, normal)
    assert abs((hole - center) @ normal) < 1e-9
    assert hole[1] == pytest.approx(center[1] + 0.02)
    assert hole[2] == pytest.approx(center[2] - 0.01)


def test_project_to_plane():
    det = BoardDetector(cam=None, target_mode="offset")
    center = np.array([0.6, 0.0, 0.3])
    normal = np.array([1.0, 0.0, 0.0])
    p = np.array([0.55, 0.02, 0.29])
    q = det._project_to_plane(p, center, normal)
    assert abs((q - center) @ normal) < 1e-9
    assert q[1] == pytest.approx(0.02)
    assert q[2] == pytest.approx(0.29)


def test_contour_circularity_circle_vs_bar():
    img = np.zeros((80, 80), np.uint8)
    cv2.circle(img, (40, 40), 20, 255, -1)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circ, _ = contour_circularity(cnts[0])
    assert circ > 0.85
    img2 = np.zeros((80, 80), np.uint8)
    cv2.rectangle(img2, (5, 36), (75, 44), 255, -1)
    cnts, _ = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circ, _ = contour_circularity(cnts[0])
    assert circ < 0.55


def test_intersect_ray_plane():
    origin = np.array([0.26, 0.02, 0.34])
    direction = np.array([1.0, 0.0, 0.0])
    center = np.array([0.64, -0.32, 0.33])
    normal = np.array([1.0, 0.0, 0.0])
    hit = intersect_ray_plane(origin, direction, center, normal)
    assert hit[0] == pytest.approx(0.64)
    assert hit[1] == pytest.approx(0.02)
    assert hit[2] == pytest.approx(0.34)


def test_on_white_pad_rejects_dark_without_ring():
    white = np.zeros((80, 80), np.uint8)
    assert BoardDetector._on_white_pad(white, 40, 40, 8) is False
    cv2.circle(white, (40, 40), 18, 255, -1)
    assert BoardDetector._on_white_pad(white, 40, 40, 8) is True


def test_planar_miss_is_in_plane_offset():
    n = np.array([1.0, 0.0, 0.0])
    hole = np.array([0.64, -0.32, 0.33])
    tip = hole - 0.03 * n + np.array([0.0, 0.01, 0.0])
    assert _planar_miss(tip, n, hole, n, 0.35) == pytest.approx(0.01, abs=1e-4)


class _FakeCam:
    fx = 469.0

    def ray_plane(self, u, v, center, normal, cam_pose):
        cx, cy = 319.5, 179.5
        dist = 0.38
        dy = (u - cx) / self.fx * dist
        dz = -(v - cy) / self.fx * dist
        return np.asarray(center, float) + np.array([0.0, dy, dz])


def test_locate_target_locks_circle_on_white_not_stray_blob():
    rgb = np.full((360, 640, 3), (204, 163, 107), np.uint8)
    rgb[155:205, 295:345] = (242, 242, 242)
    cv2.circle(rgb, (320, 180), 10, (10, 10, 10), -1)
    # stray dark bar far from the pad (old detector would score this)
    rgb[40:50, 20:120] = (15, 15, 15)
    det = BoardDetector(_FakeCam(), target_mode="marker")
    center = np.array([0.628, 0.0, 0.0])
    normal = np.array([1.0, 0.0, 0.0])
    cam_pose = (np.array([0.25, 0.0, 0.0]), np.eye(3))
    hole, src, px = det.locate_target(rgb, np.zeros((360, 640), np.float32),
                                      cam_pose, center, normal)
    assert src == "marker"
    assert px[0] == pytest.approx(320, abs=3)
    assert px[1] == pytest.approx(180, abs=3)
    assert abs(hole[1]) < 0.005
    assert abs(hole[2]) < 0.005
