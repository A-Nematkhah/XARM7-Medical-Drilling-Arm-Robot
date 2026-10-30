"""Kinematic Cartesian-servo predictor (no physics rollout)."""
import numpy as np
import pytest

from hw.defaults import REST_Q


def test_predict_cart_servo_nearby_pose():
    mujoco = pytest.importorskip("mujoco")
    import scene
    from robot import Robot

    model, _ = scene.build_model()
    data = mujoco.MjData(model)
    data.qpos[:7] = REST_Q
    mujoco.mj_forward(model, data)
    robot = Robot(model, data)
    robot.q_cmd = REST_Q.copy()
    p, R = robot.site_pose()
    tgt = p + np.array([0.0, 0.0, 0.04])
    ok, traj, err, tag = robot.predict_cart_servo(
        REST_Q, [tgt], R, hops=[0.02], tol=0.02)
    assert ok, (err, tag)
    assert err < 0.025
    assert len(traj) >= 2


def test_arm_clearance_at_rest_not_in_table():
    mujoco = pytest.importorskip("mujoco")
    import scene
    from robot import Robot

    model, _ = scene.build_model()
    data = mujoco.MjData(model)
    data.qpos[:7] = REST_Q
    mujoco.mj_forward(model, data)
    robot = Robot(model, data)
    clr, pair = robot.arm_clearance(REST_Q)
    assert clr >= 0.0
    # Rest pose should not be buried in the table.
    assert clr > 0.01, pair
