import csv
import json
from pathlib import Path

import pytest

from calibration import CalibConfig, MilestoneCalib, load_calib


ROOT = Path(__file__).resolve().parents[1]


def _write_metrics(path: Path, n_ok: int, n_fail: int = 0, tilt=0.0, azim=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["success", "seed", "tilt_deg", "azim_deg", "fail_reason"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i in range(n_ok):
            w.writerow(dict(success="1", seed=i, tilt_deg=tilt, azim_deg=azim, fail_reason=""))
        for i in range(n_fail):
            w.writerow(dict(success="0", seed=100 + i, tilt_deg=tilt, azim_deg=azim,
                            fail_reason="boom"))


def test_strict_yaml_allow_angled_does_not_unlock():
    cfg = CalibConfig(milestone=MilestoneCalib(
        allow_angled=True, strict=True,
        unlock_file="config/angled_unlocked_missing_for_test"))
    assert cfg.angled_unlocked() is False
    assert cfg.angled_unlocked(cli_allow=True) is True


def test_non_strict_yaml_allow_angled_unlocks():
    cfg = CalibConfig(milestone=MilestoneCalib(allow_angled=True, strict=False))
    assert cfg.angled_unlocked() is True


def test_unlock_file_still_unlocks(tmp_path):
    unlock = tmp_path / "angled_unlocked"
    unlock.write_text('{"success_rate": 0.9}\n', encoding="utf-8")
    cfg = CalibConfig(milestone=MilestoneCalib(unlock_file=str(unlock), strict=True))
    assert cfg.angled_unlocked() is True


def test_unlock_script_writes_stamp(tmp_path):
    metrics = tmp_path / "eval_batch.csv"
    unlock = tmp_path / "angled_unlocked"
    _write_metrics(metrics, n_ok=8, n_fail=2)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "unlock_angled", ROOT / "scripts" / "unlock_angled.py")
    ua = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ua)

    stamp = ua.evaluate_baseline(metrics, min_success=0.8, min_runs=10)
    assert stamp["success_rate"] == pytest.approx(0.8)
    unlock.write_text(json.dumps(stamp), encoding="utf-8")
    assert json.loads(unlock.read_text(encoding="utf-8"))["n"] == 10


def test_unlock_script_rejects_low_rate(tmp_path):
    metrics = tmp_path / "eval_batch.csv"
    _write_metrics(metrics, n_ok=5, n_fail=5)
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "unlock_angled", ROOT / "scripts" / "unlock_angled.py")
    ua = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ua)

    with pytest.raises(RuntimeError, match="success"):
        ua.evaluate_baseline(metrics, min_success=0.8, min_runs=10)


def test_default_calib_loads_hardware_and_strict_milestone():
    cfg = load_calib(ROOT / "config" / "calib.yaml")
    assert cfg.hardware.arm_ip
    assert cfg.milestone.strict is True
    assert cfg.milestone.min_perp_success == pytest.approx(0.8)
    cfg.milestone.unlock_file = "config/angled_unlocked_missing_for_test"
    assert cfg.angled_unlocked() is False
