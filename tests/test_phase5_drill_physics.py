"""Phase 5: dry-cut abort, chuck contact force, camera latency, GT is report-only."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from safety import GateError, SafetyGates

from calibration import load_calib


ROOT = Path(__file__).resolve().parents[1]


def test_dry_cut_aborts_without_motor():
    g = SafetyGates()
    g.check_dry_cut(0.001, motor_on=False, phase="ok")
    g.check_dry_cut(0.05, motor_on=True, phase="feed")  # spinning: allowed
    with pytest.raises(GateError, match="dry cut"):
        g.check_dry_cut(0.005, motor_on=False, phase="approach")


def test_board_contact_force_gate():
    g = SafetyGates()
    g.check_board_contact_force(10.0, "ok")
    with pytest.raises(GateError, match="contact force"):
        g.check_board_contact_force(40.0, "7/8")


def test_gates_yaml_loads_dry_cut_limits():
    g = SafetyGates.from_file()
    assert g.MAX_DRY_PEN == pytest.approx(0.002)
    assert g.MAX_BOARD_CONTACT_N == pytest.approx(25.0)
    assert g.CHUCK_BEHIND_TIP == pytest.approx(0.094)
    assert g.CHUCK_R == pytest.approx(0.018)


def test_bit_geoms_are_noncolliding_chuck_is_not():
    import mujoco
    import scene

    model, _ = scene.build_model()
    bit = model.geom("bit")
    chuck = model.geom("chuck")
    assert int(model.geom_contype[bit.id]) == 0
    assert int(model.geom_conaffinity[bit.id]) == 0
    assert int(model.geom_contype[chuck.id]) != 0
    # No chuck↔board exclude in the compiled model.
    # exclude pairs are in model.pair_*; simpler: scene no longer calls add_exclude.
    src = (ROOT / "sim" / "scene.py").read_text(encoding="utf-8")
    assert "add_exclude" not in src or "chuck" not in src.split("add_exclude")[-1][:80]


def test_no_chuck_board_exclude_in_scene_source():
    text = (ROOT / "sim" / "scene.py").read_text(encoding="utf-8")
    assert "bodyname1=\"chuck\"" not in text
    assert "bodyname1='chuck'" not in text


def test_camera_latency_returns_older_frame():
    from perception import D435i

    class _Cam(D435i):
        def __init__(self):
            self.noise = False
            self.min_z, self.max_z = 0.0, 10.0
            self.dropout = 0.0
            self.edge_noise_m = 0.0
            self._latency_frames = 2
            self._frame_buf = deque(maxlen=3)
            self._n = 0
            self.cam = 0
            self.d = None
            self.rgb_r = SimpleNamespace(update_scene=lambda *a, **k: None,
                                         render=lambda: self._rgb())
            self.dep_r = SimpleNamespace(update_scene=lambda *a, **k: None,
                                         render=lambda: self._dep(),
                                         enable_depth_rendering=lambda: None)

        def _rgb(self):
            img = np.full((4, 4, 3), self._n, np.uint8)
            return img

        def _dep(self):
            z = np.full((4, 4), 0.5 + 0.01 * self._n, np.float32)
            self._n += 1
            return z

    cam = _Cam()
    r0, _ = cam.capture()  # warm-up → newest
    r1, _ = cam.capture()
    r2, _ = cam.capture()  # buffer full: should return frame 0
    assert int(r0[0, 0, 0]) == 0
    assert int(r1[0, 0, 0]) == 1
    assert int(r2[0, 0, 0]) == 0  # delayed


def test_settle_capture_discards_motion_tainted_buffer():
    from perception import D435i

    class _Cam(D435i):
        def __init__(self):
            self.noise = False
            self.min_z, self.max_z = 0.0, 10.0
            self.dropout = 0.0
            self.edge_noise_m = 0.0
            self._latency_frames = 2
            self._frame_buf = deque(maxlen=3)
            self._n = 0
            self.cam = 0
            self.d = None
            self.rgb_r = SimpleNamespace(update_scene=lambda *a, **k: None,
                                         render=lambda: self._rgb())
            self.dep_r = SimpleNamespace(update_scene=lambda *a, **k: None,
                                         render=lambda: self._dep(),
                                         enable_depth_rendering=lambda: None)

        def _rgb(self):
            return np.full((4, 4, 3), self._n, np.uint8)

        def _dep(self):
            z = np.full((4, 4), 0.5, np.float32)
            self._n += 1
            return z

    cam = _Cam()
    for _ in range(3):
        cam.capture()  # buffer holds 0..2, returns 0
    assert cam._n == 3
    rgb, _ = cam.settle_capture()  # flush + 3 new frames (3,4,5); returns 3
    assert int(rgb[0, 0, 0]) == 3
    assert cam._n == 6


def test_gt_metrics_are_not_safety_gates():
    """entry_err_vs_gt / depth_gt must never appear in safety.py gate logic."""
    text = (ROOT / "sim" / "safety.py").read_text(encoding="utf-8")
    assert "entry_err_vs_gt" not in text
    assert "depth_gt" not in text
    assert "BOARD_POS" not in text or "not MuJoCo BOARD_POS" in text


def test_camera_calib_defaults_loaded():
    cfg = load_calib(ROOT / "config" / "calib.yaml")
    assert cfg.camera.latency_s == pytest.approx(0.067)
    assert cfg.camera.dropout == pytest.approx(0.02)
