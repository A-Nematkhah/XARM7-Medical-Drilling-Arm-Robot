import numpy as np
import pytest

from task.grasp_ctrl import (
    CLOSE_TIMEOUT, LOADED_IDX, close_until_load, n_loaded, overlay_index,
)


class FakeHand:
    def __init__(self, angles=None, force_at=400.0, load=0.5):
        self.angles = np.array(angles if angles is not None else [1000.0] * 6, float)
        self.target = self.angles.copy()
        self.force_at = float(force_at)
        self.load = float(load)
        self.history = []

    def set_angles(self, angles, speed=None):
        del speed
        self.target = np.clip(np.asarray(angles, float), 0.0, 1000.0)
        self.history.append(self.target.copy())

    def get_angles(self):
        return self.angles.copy()

    def get_forces(self):
        f = np.zeros(6)
        for i in LOADED_IDX:
            if self.angles[i] <= self.force_at:
                f[i] = self.load
        return f


class FakeRobot:
    def __init__(self, hand, step=40.0):
        self.hand = hand
        self.step = float(step)
        self.waits = 0

    def wait(self, t):
        del t
        self.waits += 1
        h = self.hand
        delta = np.clip(h.target - h.angles, -self.step, self.step)
        h.angles = np.clip(h.angles + delta, 0.0, 1000.0)


def test_n_loaded_counts_wrap_fingers():
    assert n_loaded([0.4, 0.5, 0.5, 0.0, 0.4, 0.0], 0.4) == 4
    assert n_loaded([0.1, 0.1, 0.1, 0.0, 0.1, 0.0], 0.4) == 0


def test_close_stops_when_loaded_and_freezes_command():
    hand = FakeHand()
    robot = FakeRobot(hand)
    r = close_until_load(
        hand, robot, target=[0, 0, 0, 1000, 0, 0],
        load=0.4, min_loaded=3, loaded_idx=LOADED_IDX,
        dt=0.05, timeout=2.5, settle=0.0, speed=800)
    assert r.n_loaded >= 3
    assert r.elapsed < 2.6
    # Frozen commands stay near contact, not slammed all the way to 0.
    assert r.cmd[0] > 100.0
    assert r.cmd[4] > 100.0
    assert any(r.frozen)
    assert hand.target[0] == pytest.approx(r.cmd[0])
    assert hand.target[3] == pytest.approx(1000.0)  # index stays open


def test_already_loaded_does_not_keep_closing():
    hand = FakeHand(angles=[300, 300, 300, 1000, 300, 0])
    robot = FakeRobot(hand)
    r = close_until_load(
        hand, robot, target=[0, 0, 0, 1000, 0, 0],
        load=0.4, min_loaded=3, dt=0.05, timeout=2.5, settle=0.0, speed=800,
        hold_scale=1.0, hold_extra=0.0)
    assert r.n_loaded >= 3
    assert robot.waits == 0
    assert r.cmd[0] == pytest.approx(300.0)


def test_timeout_returns_without_raising():
    hand = FakeHand(force_at=-1.0)  # never contacts
    robot = FakeRobot(hand)
    r = close_until_load(
        hand, robot, target=[0, 0, 0, 1000, 0, 0],
        load=0.4, min_loaded=3, dt=0.05, timeout=0.2, settle=0.0, speed=800)
    assert r.n_loaded == 0
    assert r.elapsed == pytest.approx(0.2)
    assert r.cmd[0] < 1000.0


def test_close_timeout_gives_more_than_two_seconds():
    assert CLOSE_TIMEOUT >= 3.0


def test_overlay_index_keeps_wrap_fingers():
    cmd = overlay_index([320.0, 310.0, 300.0, 1000.0, 280.0, 0.0], 0)
    assert cmd[0] == pytest.approx(320.0)
    assert cmd[3] == pytest.approx(0.0)
    assert cmd[4] == pytest.approx(280.0)
    opened = overlay_index(cmd, 1000)
    assert opened[0] == pytest.approx(320.0)
    assert opened[3] == pytest.approx(1000.0)


def test_index_and_thumb_rot_snap_to_target():
    hand = FakeHand(angles=[1000, 1000, 1000, 500, 1000, 250])
    robot = FakeRobot(hand)
    r = close_until_load(
        hand, robot, target=[0, 0, 0, 1000, 0, 0],
        load=0.4, min_loaded=3, dt=0.05, timeout=2.5, settle=0.0, speed=800)
    assert r.cmd[3] == pytest.approx(1000.0)
    assert r.cmd[5] == pytest.approx(0.0)
