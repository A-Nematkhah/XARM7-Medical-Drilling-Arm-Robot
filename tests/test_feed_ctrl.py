import pytest

from task.feed_ctrl import DepthFeedCtrl
from task.motion_profiles import load_motion_profiles


def test_large_error_commands_near_vmax():
    c = DepthFeedCtrl(target=0.025, vmax=0.012, kp=0.55, ki=0.0, vmin=0.0025, dn=1.0)
    step, speed = c.command(pen_m=0.0, remaining_along=0.05, dt=0.22)
    assert speed == pytest.approx(0.012)  # 0.55*0.025=0.01375 → cap
    assert 0.0 < step <= 0.004


def test_near_target_slows_down():
    c = DepthFeedCtrl(target=0.025, vmax=0.012, kp=0.55, ki=0.0, vmin=0.0025,
                      max_step=0.004, tol=0.001, dn=1.0)
    _, v_far = c.command(0.000, 0.05, 0.22)
    c.i_err = 0.0
    _, v_mid = c.command(0.0215, 0.05, 0.22)  # 3.5 mm left → vmin floor
    c.i_err = 0.0
    _, v_near = c.command(0.0232, 0.05, 0.22)  # 1.8 mm left → crawl below vmin
    assert v_mid == pytest.approx(0.0025)
    assert v_near < v_mid
    assert v_near < v_far


def test_at_target_stops():
    c = DepthFeedCtrl(target=0.025, vmax=0.012, tol=0.001)
    step, speed = c.command(pen_m=0.0245, remaining_along=0.05, dt=0.22)
    assert step == 0.0 and speed == 0.0


def test_remaining_budget_clips_step():
    c = DepthFeedCtrl(target=0.025, vmax=0.012, kp=0.55, ki=0.0, max_step=0.004, dn=1.0)
    step, _ = c.command(pen_m=0.0, remaining_along=0.001, dt=0.22)
    assert step == pytest.approx(0.001)


def test_from_profile_reads_yaml_gains():
    book = load_motion_profiles()
    ctrl = DepthFeedCtrl.from_profile(
        book["feed"], target=0.025, vmax=book["feed"].vmax, dn=1.0, tol=0.001)
    assert ctrl.kp == pytest.approx(0.55)
    assert ctrl.vmax == pytest.approx(0.012)
    assert ctrl.max_step == pytest.approx(0.004)
    assert ctrl.f_nom == pytest.approx(3.5)
    assert ctrl.f_hi == pytest.approx(8.0)
    assert ctrl.rpm_nom == pytest.approx(1100.0)
    assert ctrl.rpm_lo == pytest.approx(400.0)
    assert ctrl.load_scale_min == pytest.approx(0.25)


def _loaded():
    return DepthFeedCtrl(
        target=0.025, vmax=0.012, kp=0.55, ki=0.0, vmin=0.0025,
        max_step=0.004, tol=0.001, dn=1.0,
        f_nom=3.5, f_hi=8.0, rpm_nom=1100.0, rpm_lo=400.0, load_scale_min=0.25)


def test_missing_sensors_do_not_derate():
    c = _loaded()
    assert c.load_scale(0.0, None) == pytest.approx(1.0)
    assert c.load_scale(0.0, None, motor_on=True) == pytest.approx(1.0)
    _, v0 = c.command(0.0, 0.05, 0.22, force_n=0.0, rpm=None, motor_on=True)
    assert v0 == pytest.approx(0.012)
    # rpm=0 with motor off is idle, not a stall.
    assert c.load_scale(0.0, 0.0) == pytest.approx(1.0)


def test_motor_on_zero_rpm_derates_as_stall():
    c = _loaded()
    assert c.load_scale(0.0, 0.0, motor_on=True) == pytest.approx(0.25)
    _, v = c.command(0.0, 0.05, 0.22, force_n=0.0, rpm=0.0, motor_on=True)
    assert v == pytest.approx(0.012 * 0.25)


def test_high_thrust_derates_vmax():
    c = _loaded()
    # 6 N → 1 - (6-3.5)/(8-3.5) = 0.444... → vmax 5.33 mm/s
    s = c.load_scale(6.0, 1400.0)
    assert s == pytest.approx(1.0 - (6.0 - 3.5) / (8.0 - 3.5))
    _, v = c.command(0.0, 0.05, 0.22, force_n=6.0, rpm=1400.0)
    assert v == pytest.approx(0.012 * s)
    assert v < 0.012


def test_low_rpm_derates_vmax():
    c = _loaded()
    s = c.load_scale(0.0, 750.0)  # (750-400)/(1100-400) = 0.5
    assert s == pytest.approx(0.5)
    _, v = c.command(0.0, 0.05, 0.22, force_n=0.0, rpm=750.0)
    assert v == pytest.approx(0.006)


def test_load_scale_takes_the_stricter_axis():
    c = _loaded()
    s_f = c.load_scale(8.0, 1400.0)   # floor
    s_w = c.load_scale(0.0, 400.0)    # floor
    s_both = c.load_scale(8.0, 400.0)
    assert s_f == pytest.approx(0.25)
    assert s_w == pytest.approx(0.25)
    assert s_both == pytest.approx(0.25)


def test_idle_rpm_does_not_derate():
    c = _loaded()
    assert c.load_scale(0.0, 1432.0) == pytest.approx(1.0)
