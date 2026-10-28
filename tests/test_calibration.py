from pathlib import Path

import numpy as np
import pytest

from calibration import (
    BoardCalib, CalibConfig, HandEyeCalib, camera_pose_from_tcp, hole_from_prior,
    load_calib, on_board_face, rpy_deg_to_R, tcp_from_camera,
)


ROOT = Path(__file__).resolve().parents[1]


def test_load_default_calib_yaml():
    cfg = load_calib(ROOT / "config" / "calib.yaml")
    assert cfg.source.endswith("calib.yaml")
    assert cfg.board.normal[0] == pytest.approx(1.0)
    assert np.allclose(cfg.board.hole_yz, [0.02, -0.01])
    assert cfg.board.half[0] == pytest.approx(0.012)
    assert np.allclose(cfg.hand_eye.t, [-0.0254, 0.072, -0.0975])
    assert np.allclose(cfg.hand_eye.rpy_deg, [0.0, -90.0, 0.0])
    assert cfg.tool.bit_reach == pytest.approx(0.2075)
    assert cfg.tool.housing_to_tip == pytest.approx(0.2115)
    assert cfg.tool.held_bit_reach == pytest.approx(0.202)
    assert cfg.tool.chuck_behind_tip == pytest.approx(0.094)
    assert cfg.hand.force_scale == pytest.approx(1.0)
    assert cfg.camera.min_z == pytest.approx(0.105)
    assert cfg.milestone.allow_angled is False


def test_on_board_face_uses_detected_hole_and_calib_half():
    hole = np.array([0.628, -0.32, 0.33])
    n = np.array([1.0, 0.0, 0.0])
    half = np.array([0.012, 0.16, 0.13])
    assert on_board_face(hole, hole, n, half)
    assert on_board_face(hole + np.array([0.0, 0.05, 0.0]), hole, n, half)
    assert not on_board_face(hole + np.array([0.0, 0.40, 0.0]), hole, n, half)


def test_hole_from_prior_matches_axis_aligned_board():
    board = BoardCalib(
        position=np.array([0.64, -0.34, 0.34]),
        normal=np.array([1.0, 0.0, 0.0]),
        hole_yz=np.array([0.02, -0.01]),
        half=np.array([0.012, 0.16, 0.13]),
    )
    hole = hole_from_prior(board)
    assert np.allclose(hole, [0.628, -0.32, 0.33])


def test_rpy_minus_90_pitch_is_optical_mount():
    R = rpy_deg_to_R([0.0, -90.0, 0.0])
    assert np.allclose(R, [[0, 0, -1], [0, 1, 0], [1, 0, 0]])


def test_camera_pose_roundtrip_tcp():
    he = HandEyeCalib()
    p_ee = np.array([0.4, 0.0, 0.3])
    R_ee = np.eye(3)
    p_cam, R_cam = camera_pose_from_tcp(p_ee, R_ee, he)
    p2, R2 = tcp_from_camera(p_cam, R_cam, he)
    assert np.allclose(p2, p_ee)
    assert np.allclose(R2, R_ee)


def test_angled_unlocked_via_cli_flag():
    cfg = CalibConfig()
    assert cfg.angled_unlocked(cli_allow=True) is True
    assert cfg.angled_unlocked(cli_allow=False) is False


def test_angled_unlocked_via_file(tmp_path, monkeypatch):
    unlock = tmp_path / "angled_unlocked"
    unlock.write_text("", encoding="utf-8")
    cfg = CalibConfig()
    cfg.milestone.unlock_file = str(unlock)
    assert cfg.angled_unlocked() is True
