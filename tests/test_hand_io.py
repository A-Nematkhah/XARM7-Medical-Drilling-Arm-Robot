from types import SimpleNamespace

import numpy as np

from task.hand_io import (
    index_angle, scale_finger_forces, trigger_physically_engaged,
    trigger_pressed, trigger_released,
)


def test_trigger_pressed_from_index_angle():
    hand = SimpleNamespace(get_angles=lambda: [0, 0, 0, 50, 0, 0])
    assert trigger_pressed(hand)
    assert not trigger_released(hand)
    assert index_angle(hand) == 50
    # Contact pad often leaves the encoder around 400–420; ON must cover that.
    from task.constants import INDEX_TRIGGER_ON
    assert INDEX_TRIGGER_ON >= 450
    hand408 = SimpleNamespace(get_angles=lambda: [0, 0, 0, 408, 0, 0])
    assert trigger_pressed(hand408)


def test_trigger_released_when_index_extended():
    hand = SimpleNamespace(get_angles=lambda: [0, 0, 0, 1000, 0, 0])
    assert not trigger_pressed(hand)
    assert trigger_released(hand)


def test_index_falls_back_to_target():
    hand = SimpleNamespace(
        get_angles=lambda: (_ for _ in ()).throw(RuntimeError("no encoder")),
        target=[0, 0, 0, 0, 0, 0],
    )
    assert trigger_pressed(hand)


def test_physical_trigger_true_without_sim_model():
    ctx = SimpleNamespace(model=None, data=None, trig_q=-1)
    assert trigger_physically_engaged(ctx) is True


def test_physical_trigger_reads_sim_qpos():
    ctx = SimpleNamespace(
        model=object(),
        data=SimpleNamespace(qpos=np.array([0.0, 0.002])),
        trig_q=1,
    )
    assert trigger_physically_engaged(ctx) is True
    ctx.data.qpos[1] = 0.0
    assert trigger_physically_engaged(ctx) is False


def test_scale_finger_forces():
    assert np.allclose(scale_finger_forces([400, 0], 0.001), [0.4, 0.0])
