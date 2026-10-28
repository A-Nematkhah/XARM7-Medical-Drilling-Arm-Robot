"""Phase 4: task talks only through ArmDriver; hardware path has no mj_jacSite."""
from pathlib import Path

import numpy as np
import pytest

from hw.defaults import REST_Q
from hw.interfaces import ArmDriver
from hw.real_backend import RealArm, RealBackendNotReady, make_real_backend


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "task"
HW = ROOT / "hw"


import re

# Live-loop cheats: mj_jacSite / servo_step / peeking into ArmIK. Planning
# helpers like ik_solutions / plan_path_ik stay on ArmDriver.
FORBIDDEN_TASK = re.compile(
    r"(?<![A-Za-z_])servo_step(?![A-Za-z_])"
    r"|(?<![A-Za-z_])mj_jacSite(?![A-Za-z_])"
    r"|robot\.ik(?![A-Za-z_])"
    r"|\.ik\.qadr"
    r"|from robot import REST_Q"
)
FORBIDDEN_REAL_CODE = re.compile(
    r"^\s*(import mujoco|from mujoco|.*\bmj_jacSite\s*\(|.*\bservo_step\s*\(|.*\bArmIK\b)",
    re.M,
)


def test_task_package_never_calls_live_ik_servo():
    hits = []
    for path in TASK.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for m in FORBIDDEN_TASK.finditer(text):
            hits.append(f"{path.relative_to(ROOT)}: {m.group(0)}")
    assert hits == []


def test_real_backend_has_no_mujoco_or_jac():
    hits = []
    for name in ("real_backend.py", "interfaces.py", "defaults.py"):
        text = (HW / name).read_text(encoding="utf-8")
        for m in FORBIDDEN_REAL_CODE.finditer(text):
            hits.append(f"hw/{name}: {m.group(0).strip()}")
    assert hits == []


def test_real_arm_is_arm_driver_and_blocks_motion():
    arm = RealArm(arm_ip="127.0.0.1")
    assert isinstance(arm, ArmDriver)
    assert np.allclose(arm.rest_q, REST_Q)
    assert np.allclose(arm.get_q(), REST_Q)
    with pytest.raises(RealBackendNotReady, match="move_to_pose"):
        arm.move_to_pose(np.zeros(3), np.eye(3))
    with pytest.raises(RealBackendNotReady, match="move_joints"):
        arm.move_joints(REST_Q)
    with pytest.raises(RealBackendNotReady, match="goto_pose"):
        arm.goto_pose_joint_space(np.zeros(3), np.eye(3))


def test_make_real_backend_returns_arm_driver():
    be = make_real_backend()
    assert isinstance(be.arm, ArmDriver)
    with pytest.raises(RealBackendNotReady):
        be.hand.set_angles(be.hand.OPEN)


def test_run_drill_task_does_not_attach_sim_callbacks_on_real():
    text = (ROOT / "run_drill_task.py").read_text(encoding="utf-8")
    assert "attach_sim_callbacks(ctx)" in text
    # Real path must not build a MuJoCo scene or attach sim-only motor/contact.
    assert 'if args.backend == "real":' in text
    assert "model = data = None" in text
    assert "attach_sim_callbacks(ctx)" in text.split('if args.backend == "sim":')[-1]


def test_sim_robot_exposes_get_q_and_qadr():
    import mujoco
    import scene
    from robot import Robot, REST_Q as SIM_REST

    model, _ = scene.build_model()
    data = mujoco.MjData(model)
    data.qpos[:7] = SIM_REST
    mujoco.mj_forward(model, data)
    arm = Robot(model, data)
    assert isinstance(arm, ArmDriver)
    assert arm.qadr.shape == (7,)
    assert np.allclose(arm.get_q(), SIM_REST)
    assert arm.limit_margin(arm.get_q()) > 0.1
