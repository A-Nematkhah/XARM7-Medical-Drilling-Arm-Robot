from pathlib import Path

import pytest

from task.motion_profiles import MotionBook, MotionProfile, load_motion_profiles


ROOT = Path(__file__).resolve().parents[1]


def test_load_default_motion_yaml():
    book = load_motion_profiles(ROOT / "config" / "motion.yaml")
    assert book.source.endswith("motion.yaml")
    assert set(book.names()) >= {
        "scan", "grasp", "lift", "relocate", "approach", "feed", "retract", "place"}
    assert book["approach"].vmax < book["relocate"].vmax
    assert book["feed"].vmax <= book["retract"].vmax
    assert book["feed"].vmax < book["approach"].vmax
    assert book["feed"].kp > 0.0
    assert book["feed"].max_step > 0.0
    assert book["feed"].f_hi > book["feed"].f_nom > 0.0
    assert book["feed"].rpm_nom > book["feed"].rpm_lo > 0.0


def test_feed_clamped_to_gate_cap():
    book = load_motion_profiles(ROOT / "config" / "motion.yaml", feed_vmax_cap=0.008)
    assert book["feed"].vmax == pytest.approx(0.008)


def test_duration_uses_vmax_and_t_min():
    p = MotionProfile(name="x", vmax=0.04, t_min=0.5, settle=0.1)
    assert p.duration(0.01) == pytest.approx(0.5)          # t_min wins
    assert p.duration(0.12) == pytest.approx(3.0)          # dist/vmax
    kw = p.cart_kwargs(0.12)
    assert kw["T"] == pytest.approx(3.0)
    assert kw["vmax"] == pytest.approx(0.04)
    assert kw["settle"] == pytest.approx(0.1)


def test_unknown_profile_raises():
    book = MotionBook({"scan": MotionProfile("scan", 0.1, 1.0)})
    with pytest.raises(KeyError, match="unknown"):
        _ = book["feed"]
