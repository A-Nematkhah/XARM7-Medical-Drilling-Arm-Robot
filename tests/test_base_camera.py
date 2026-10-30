"""Column RGB-D is gone; LiDAR lives on the hand."""
import numpy as np

from hw.real_backend import make_real_backend


def test_scene_has_no_column_camera():
    import scene
    model, _ = scene.build_model()
    names = {model.camera(i).name for i in range(model.ncam)}
    assert "base_d435i_color" not in names
    bodies = {model.body(i).name for i in range(model.nbody)}
    assert "base_d435i" not in bodies
    assert "base_lidar" not in bodies
    assert "wrist_lidar" in bodies
    assert "d435i" in bodies


def test_sim_backend_has_wrist_lidar_not_base_camera():
    import mujoco
    import scene
    from calibration import load_calib
    from hw.sim_backend import make_sim_backend
    from lidar import WristLidar
    from robot import REST_Q

    cfg = load_calib()
    model, _ = scene.build_model()
    data = mujoco.MjData(model)
    data.qpos[:7] = REST_Q
    mujoco.mj_forward(model, data)
    be = make_sim_backend(model, data, noise=False, hand_eye=cfg.hand_eye,
                          lidar_eye=cfg.lidar_eye)
    assert not hasattr(be, "base_camera") or getattr(be, "base_camera", None) is None
    assert isinstance(be.lidar, WristLidar)
    p, R = be.lidar.pose()
    bid = int(model.body("wrist_lidar").id)
    assert np.linalg.norm(p - data.xpos[bid]) < 1e-3
    assert float(R[2, 2]) > 0.98


def test_real_backend_has_no_base_camera():
    be = make_real_backend()
    assert not hasattr(be, "base_camera") or getattr(be, "base_camera", None) is None
