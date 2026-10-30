"""Wrist 360° LiDAR: clustering + sim rays, no MuJoCo board constants in task."""
import numpy as np
import pytest

from lidar import locate_objects
from task.viz import render_lidar_image


def test_locate_objects_splits_compact_drill_and_long_board():
    drill = np.array([[0.40 + 0.01 * np.cos(t), 0.25 + 0.01 * np.sin(t), 0.10]
                      for t in np.linspace(0, 2 * np.pi, 24)])
    # Tall vertical face (Z extent) + long Y wall — not a flat table blob.
    board = np.array([[0.64, -0.34 + y, 0.22 + z]
                      for y in np.linspace(-0.12, 0.12, 24)
                      for z in np.linspace(0.0, 0.24, 8)])
    cmap = locate_objects(np.vstack([drill, board]), origin_xy=(0.0, 0.0))
    assert cmap.drill_xy is not None
    assert cmap.board_xy is not None
    assert np.linalg.norm(cmap.drill_xy - [0.40, 0.25]) < 0.04
    assert np.linalg.norm(cmap.board_xy - [0.64, -0.34]) < 0.05
    assert abs(cmap.board_n[2]) < 0.2
    # Into-board points away from the robot.
    assert float(cmap.board_n[:2] @ cmap.board_xy) > 0


def test_locate_objects_does_not_swap_drill_and_board():
    """Low flat blob = drill; tall thin wall = board — never swapped."""
    drill = np.array([[0.30 + 0.02 * np.cos(t), 0.20 + 0.015 * np.sin(t), 0.08 + 0.01 * np.sin(2 * t)]
                      for t in np.linspace(0, 2 * np.pi, 30)])
    board = np.array([[0.65, y, z]
                      for y in np.linspace(-0.15, 0.15, 28)
                      for z in np.linspace(0.20, 0.42, 10)])
    cmap = locate_objects(np.vstack([drill, board]), origin_xy=(0.1, 0.0))
    assert cmap.drill_xy is not None and cmap.board_xy is not None
    assert np.linalg.norm(cmap.drill_xy - [0.30, 0.20]) < 0.05
    assert np.linalg.norm(cmap.board_xy - [0.65, 0.0]) < 0.08


def test_locate_objects_merges_housing_and_chuck_blobs():
    """Two nearby flat blobs (housing + chuck) must become one drill, not two."""
    housing = np.array([[0.28 + 0.015 * np.cos(t), 0.25 + 0.012 * np.sin(t), 0.12]
                        for t in np.linspace(0, 2 * np.pi, 20)])
    chuck = np.array([[0.42 + 0.01 * np.cos(t), 0.20 + 0.01 * np.sin(t), 0.18]
                      for t in np.linspace(0, 2 * np.pi, 16)])
    board = np.array([[0.64, y, z]
                      for y in np.linspace(-0.14, 0.14, 24)
                      for z in np.linspace(0.20, 0.40, 8)])
    cmap = locate_objects(np.vstack([housing, chuck, board]), origin_xy=(0.1, 0.0))
    assert cmap.drill_xy is not None and cmap.board_xy is not None
    assert np.linalg.norm(cmap.drill_xy - [0.35, 0.23]) < 0.10
    assert np.linalg.norm(cmap.board_xy - [0.64, 0.0]) < 0.08
    assert np.linalg.norm(cmap.drill_xy - cmap.board_xy) > 0.20


def test_locate_objects_empty():
    cmap = locate_objects(np.full((3, 3), np.nan))
    assert cmap.drill_xy is None
    assert cmap.board_xy is None


def _tcp_fn(model, data):
    sid = int(model.site("grasp_site").id)

    def pose():
        return data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()

    return pose


def test_wrist_lidar_finds_seed2_drill_at_rest():
    """Self-filtered 360° scan sees a drill beside the palm without moving."""
    import mujoco
    import scene
    from calibration import load_calib
    from hw.sim_backend import make_sim_backend
    from robot import REST_Q

    cfg = load_calib()
    rng = np.random.default_rng(2)
    xy, yaw, board_pos, hole_yz, board_yaw = scene.sample_layout(rng)
    model, _ = scene.build_model(
        drill_xy=tuple(xy), drill_yaw=yaw,
        board_pos=board_pos, hole_yz=hole_yz, board_yaw=board_yaw)
    data = mujoco.MjData(model)
    data.qpos[:7] = REST_Q
    mujoco.mj_forward(model, data)
    be = make_sim_backend(model, data, noise=False, hand_eye=cfg.hand_eye,
                          lidar_eye=cfg.lidar_eye)
    cmap = be.lidar.locate()
    dxy = data.body("drill").xpos[:2]
    assert cmap.drill_xy is not None, "lidar missed seed-2 drill at REST"
    assert cmap.board_xy is not None, "lidar missed seed-2 board at REST"
    assert np.linalg.norm(cmap.drill_xy - dxy) < 0.15
    # Must not swap labels.
    assert np.linalg.norm(cmap.drill_xy - dxy) < np.linalg.norm(cmap.drill_xy - data.body("board").xpos[:2])
    assert np.linalg.norm(cmap.board_xy - data.body("board").xpos[:2]) < np.linalg.norm(cmap.board_xy - dxy)


def test_wrist_lidar_finds_default_scene_objects():
    import mujoco
    import scene
    from calibration import load_calib
    from lidar import WristLidar
    from robot import REST_Q

    cfg = load_calib()
    model, _ = scene.build_model()
    data = mujoco.MjData(model)
    data.qpos[:7] = REST_Q
    mujoco.mj_forward(model, data)

    lidar = WristLidar(model, data, lidar_eye=cfg.lidar_eye, tcp_pose_fn=_tcp_fn(model, data),
                       noise=False, seed=0)
    cmap = lidar.locate()
    drill_xy = data.body("drill").xpos[:2]
    board_xy = data.body("board").xpos[:2]
    assert cmap.drill_xy is not None, "lidar missed the drill"
    assert cmap.board_xy is not None, "lidar missed the board"
    assert np.linalg.norm(cmap.drill_xy - drill_xy) < 0.10
    assert np.linalg.norm(cmap.board_xy - board_xy) < 0.14


def test_wrist_lidar_pose_matches_lidar_eye():
    import mujoco
    import scene
    from calibration import camera_pose_from_tcp, load_calib
    from lidar import WristLidar
    from robot import REST_Q

    cfg = load_calib()
    model, _ = scene.build_model()
    data = mujoco.MjData(model)
    data.qpos[:7] = REST_Q
    mujoco.mj_forward(model, data)
    tcp = _tcp_fn(model, data)

    lidar = WristLidar(model, data, lidar_eye=cfg.lidar_eye, tcp_pose_fn=tcp, noise=False)
    p, R = lidar.pose()
    p2, R2 = camera_pose_from_tcp(*tcp(), cfg.lidar_eye)
    assert np.allclose(p, p2)
    assert np.allclose(R, R2)
    bid = int(model.body("wrist_lidar").id)
    assert np.linalg.norm(p - data.xpos[bid]) < 1e-3
    # Spin axis world-up at rest.
    assert float(R[2, 2]) > 0.98


def test_sample_layout_keeps_drill_off_the_board():
    import scene
    rng = np.random.default_rng(3)
    left_edge = scene.board_left_y()
    for _ in range(20):
        xy, _, pos, _, yaw = scene.sample_layout(rng)
        assert np.allclose(pos, scene.BOARD_POS)
        assert yaw == 0.0
        assert float(xy[1]) > left_edge
        assert np.all(xy >= scene.DRILL_SPAWN_LO - 1e-12)
        assert np.all(xy <= scene.DRILL_SPAWN_HI + 1e-12)


def test_real_lidar_not_ready():
    from hw.real_backend import RealBackendNotReady, RealLidar
    with pytest.raises(RealBackendNotReady):
        RealLidar(lidar_eye=None).scan()


def test_paint_lidar_scene_fills_user_geoms():
    import mujoco
    import scene
    from lidar import LidarScan
    from task.viz import paint_lidar_scene

    model, _ = scene.build_model()

    class _V:
        def __init__(self):
            self.user_scn = mujoco.MjvScene(model, maxgeom=64)

    pts = np.array([[0.4, 0.0, 0.1], [0.5, -0.2, 0.3], [np.nan, 0, 0]])
    scan = LidarScan(origin=np.array([0.2, 0.0, 0.4]), points=pts,
                     ranges=np.array([0.3, 0.4, np.inf]), n_azim=3, n_elev=1)
    v = _V()
    paint_lidar_scene(v, scan)
    assert v.user_scn.ngeom >= 3  # origin + 2 hits (+ optional rays)


def test_render_lidar_image_shape():
    from lidar import LidarScan
    n_az, n_el = 36, 4
    pts = np.full((n_az * n_el, 3), np.nan)
    rngs = np.full(n_az * n_el, np.inf)
    for i in range(12):
        pts[i] = [0.4 + 0.01 * i, 0.1, 0.12]
        rngs[i] = 0.4
    scan = LidarScan(origin=np.array([0.2, 0.0, 0.4]), points=pts, ranges=rngs,
                     n_azim=n_az, n_elev=n_el)
    img = render_lidar_image(scan, size=(320, 200))
    assert img.shape == (200, 320, 3)
    assert img.dtype == np.uint8
    assert int(img.sum()) > 0


def test_compose_task_frame_insets():
    from task.viz import compose_task_frame
    scene = np.zeros((180, 300, 3), np.uint8)
    scene[:] = (30, 30, 30)
    wrist = np.full((60, 100, 3), 200, np.uint8)
    lidar = np.full((60, 100, 3), 100, np.uint8)
    out = compose_task_frame(scene, wrist_bgr=wrist, lidar_bgr=lidar, status="test")
    assert out.shape == (180, 300, 3)
    assert out[20, 20, 0] == 100  # lidar TL
    assert out[20, 280, 0] == 200  # wrist TR
    assert int(out[-5, 20].sum()) == 0  # status bar background


def test_overlay_spawn_map_labels_box():
    from task.viz import overlay_spawn_map
    img = np.zeros((360, 640, 3), np.uint8)
    overlay_spawn_map(img, drill_xy=(0.47, -0.02))
    gray = img.reshape(-1, 3).sum(axis=1)
    assert int((gray > 0).sum()) > 100
