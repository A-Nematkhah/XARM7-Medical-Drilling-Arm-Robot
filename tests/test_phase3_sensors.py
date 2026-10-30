"""Phase 3: live sensors from calib (hand-eye, forces, geometry collision, bit reach)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safety import GateError, SafetyGates

from calibration import HandEyeCalib, camera_pose_from_tcp, load_calib
from robot import ViewPlanner, look_at_rotation
from task.hand_io import trigger_pressed


def test_view_planner_los_skips_without_model():
    he = HandEyeCalib()
    views = ViewPlanner(he, model=None)
    assert views.line_of_sight_clear(None, np.zeros(7), [np.zeros(3)], np.arange(7)) is True


def test_look_at_rotation_parallel_up_is_finite():
    R = look_at_rotation(np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    assert np.isfinite(R).all()
    assert abs(np.linalg.det(R) - 1.0) < 1e-6


def test_real_camera_project_matches_sim_shape():
    from calibration import HandEyeCalib
    from hw.real_backend import RealCamera

    cam = RealCamera(hand_eye=HandEyeCalib(),
                     tcp_pose_fn=lambda: (np.zeros(3), np.eye(3)))
    pose = (np.zeros(3), np.eye(3))
    uv, z = cam.project(np.array([[0.0, 0.0, -0.5], [0.1, 0.0, -0.5]]), cam_pose=pose)
    assert uv.shape == (2, 2)
    assert z.shape == (2,)
    assert z[0] == pytest.approx(0.5)


def test_view_planner_uses_hand_eye_translation_not_cam_xpos():
    he = HandEyeCalib(t=np.array([0.01, 0.02, 0.03]), rpy_deg=np.array([0.0, -90.0, 0.0]))
    views = ViewPlanner(he)
    cam_pos = np.array([0.30, 0.0, 0.50])
    look = np.array([0.48, 0.0, 0.0])
    p, R = views.site_target_for_view(cam_pos, look, up_hint=(0, 1, 0))
    R_exp = look_at_rotation(look - cam_pos, np.array([0.0, 1.0, 0.0]))
    assert np.allclose(R, R_exp)
    assert np.allclose(p, cam_pos - R_exp @ he.t)


def test_held_clearance_works_without_ncon():
    g = SafetyGates()
    n = np.array([1.0, 0.0, 0.0])
    hole = np.array([0.64, 0.0, 0.30])
    Rd = np.eye(3)
    tip_in_hand = np.array([0.20, 0.0, 0.0])
    p_ok = hole - 0.35 * n
    g.check_held_clearance(p_ok, Rd, tip_in_hand, hole, n, "ok")
    p_hit = hole - 0.05 * n
    with pytest.raises(GateError, match="collision"):
        g.check_held_clearance(p_hit, Rd, tip_in_hand, hole, n, "approach")


def test_d435i_pose_uses_tcp_and_calib(monkeypatch):
    from perception import D435i

    he = HandEyeCalib()
    p_ee = np.array([0.40, 0.01, 0.25])
    R_ee = np.eye(3)

    class _Stub(D435i):
        def __init__(self):
            self.hand_eye = he
            self.tcp_pose_fn = lambda: (p_ee, R_ee)
            self.d = SimpleNamespace(cam_xpos={0: np.array([9.0, 9.0, 9.0])},
                                     cam_xmat={0: np.eye(3).ravel()})
            self.cam = 0

        def mujoco_pose(self):
            raise AssertionError("live pose must not read cam_xpos")

    p, R = _Stub().pose()
    p_exp, R_exp = camera_pose_from_tcp(p_ee, R_ee, he)
    assert np.allclose(p, p_exp)
    assert np.allclose(R, R_exp)


def test_force_scale_multiplies_actuator_readout():
    class _Hand:
        aid = np.array([0, 1, 2, 3, 4, 5])
        force_scale = 10.0
        d = SimpleNamespace(actuator_force=np.array([0.1, 0.2, 0.3, 0.0, 0.4, 0.0]))

        def get_forces(self):
            from robot import InspireHand
            return InspireHand.get_forces(self)

    f = _Hand().get_forces()
    assert np.allclose(f, [1.0, 2.0, 3.0, 0.0, 4.0, 0.0])
    assert not trigger_pressed(SimpleNamespace(get_angles=lambda: [0, 0, 0, 1000, 0, 0]))


def test_hand_eye_matches_mujoco_camera_at_rest():
    import mujoco
    import scene
    from robot import REST_Q

    he = load_calib().hand_eye
    model, _ = scene.build_model()
    data = mujoco.MjData(model)
    data.qpos[:7] = REST_Q
    mujoco.mj_forward(model, data)
    p_ee = data.site_xpos[model.site("grasp_site").id]
    R_ee = data.site_xmat[model.site("grasp_site").id].reshape(3, 3)
    p_fk, R_fk = camera_pose_from_tcp(p_ee, R_ee, he)
    p_gt = data.cam_xpos[model.camera("d435i_color").id]
    R_gt = data.cam_xmat[model.camera("d435i_color").id].reshape(3, 3)
    assert np.linalg.norm(p_fk - p_gt) < 1e-4
    assert np.linalg.norm(R_fk - R_gt) < 1e-6


def test_dropout_zero_is_allowed_and_housing_to_tip_is_used():
    src = Path(__file__).resolve().parents[1] / "sim" / "perception.py"
    text = src.read_text(encoding="utf-8")
    assert "max(float(self.dropout), 0.0)" in text
    assert "c + self.housing_to_tip * ax2" in text


def test_load_calib_tool_and_hand_blocks():
    from pathlib import Path
    cfg = load_calib(Path(__file__).resolve().parents[1] / "config" / "calib.yaml")
    assert cfg.tool.bit_reach == pytest.approx(0.2075)
    assert cfg.tool.housing_to_tip == pytest.approx(0.2115)
    assert cfg.hand.force_scale == pytest.approx(1.0)
