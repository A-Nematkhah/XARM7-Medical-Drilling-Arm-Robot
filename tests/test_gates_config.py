import numpy as np
import pytest

from safety import SafetyGates, load_gate_limits, GateError


def test_yaml_limits_match_defaults():
    raw = load_gate_limits()
    assert raw["sim"]["max_tilt_deg"] == 35.0
    g = SafetyGates.from_file()
    assert g.max_tilt == pytest.approx(35.0)
    assert g.MAX_TRACK_ERR == pytest.approx(0.012)
    assert g.MIN_BOARD_PTS == 300


def test_limits_dict_overrides_track_err():
    g = SafetyGates(limits={"motion": {"max_track_err": 0.05}})
    g.check_tracking(0.04, "ok")
    with pytest.raises(GateError, match="tracking"):
        g.check_tracking(0.06, "fail")


def test_real_block_from_yaml():
    g = SafetyGates.from_file(real_robot=True)
    assert g.max_tilt == pytest.approx(20.0)
    assert g.min_dot == pytest.approx(0.75)


def test_hole_prior_rejects_wrong_blob():
    g = SafetyGates(limits={"perception": {"max_hole_prior_err": 0.04}})
    prior = np.array([0.64, -0.32, 0.33])
    g.check_hole_prior(prior + np.array([0.0, 0.01, 0.0]), prior)
    with pytest.raises(GateError, match="wrong mark"):
        g.check_hole_prior(prior + np.array([0.0, -0.09, 0.0]), prior)


def test_from_file_missing_path_raises():
    with pytest.raises(GateError, match="not found"):
        SafetyGates.from_file("config/does_not_exist.yaml")


def test_table_samples_load_from_yaml():
    g = SafetyGates.from_file()
    assert g.TABLE_SAMPLES.shape[0] >= 6
    assert g.TABLE_SAMPLES[0, 1] == pytest.approx(-0.08)


def test_hole_on_face_uses_detected_plane():
    g = SafetyGates()
    center = np.array([0.64, -0.34, 0.34])
    n = np.array([1.0, 0.0, 0.0])
    half = np.array([0.012, 0.16, 0.13])
    g.check_hole_on_face(center + np.array([0.0, 0.02, -0.01]), center, n, half)
    with pytest.raises(GateError, match="not on the detected board face"):
        g.check_hole_on_face(center + np.array([0.0, 0.80, 0.0]), center, n, half)
