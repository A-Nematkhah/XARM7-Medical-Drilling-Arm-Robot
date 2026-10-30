# Real-robot safety gates

Do not send a planned motion, trigger press, or drill feed to hardware (or treat a sim path as hardware-ready) unless it passed `sim/safety.py`. Raise `GateError` and abort; do not continue the phase.

## Must stay gated

- Drill angles: `|tilt|`, `|azim|` and `bit · board_normal`
- Grasp-site / TCP inside the workspace box (TCP-only; z min is the wrap grasp)
- Hand / LiDAR samples vs the table during transit (`check_hand_vs_table`); never joint-space-reorient near the table
- Perception confidence (grip points, radius, coarse vs fine axis)
- Board detection after grasp (plane inliers + normal facing the robot)
- Tip-to-board distance approach; refuse hand/housing plane penetration
- After grasp, never joint-space-reorient near the board (bit can ghost through; chuck contact is excluded). Retreat, then rotate, then approach.
- Joint-limit margin and Cartesian tracking error
- Grasp load + lift height before leaving the table
- Trigger only with bit standoff in front of the board
- Drill feed speed cap (mm/s), not a fixed short duration
- Chuck must keep spinning while the bit is in the face (`check_cut_spin`); dry push without motor still aborts
- Hardware feed speed must never exceed the gated / derated cap (no floor that undoes stall derate)
- On `--backend real`, `hand.force_scale` must map RH56 FORCE_ACT into N·m before grasp load is trusted

Do not aim using MuJoCo `BOARD_POS` as the live target after grasp; use `BoardDetector` (or a calibrated real board pose) and close the loop on tip-to-face clearance.

## Hardware transfer

- Do not use MuJoCo `BOARD_POS` / `CAM_POS` as ground truth on the real robot; load measured values from `config/calib.yaml`
- Wrist 360° LiDAR pose is `lidar_eye` (grasp_site / TCP). Use it only for coarse drill/board XY so the wrist D435i can look toward that region. Pierce and tip-to-face stay on the wrist D435i. Do not use a column RGB-D.
- Do not run `servo_step` IK in Python as the live xArm loop; use the controller via `hw` backends (`--backend real` stub until SDK is wired)
- Prefer `--board-target marker` (or `aruco`) over `offset` for the pierce point
- Angled holes (`tilt`/`azim` ≠ 0) stay locked until perpendicular batch eval is stable (`python scripts/unlock_angled.py` → `config/angled_unlocked`). `--allow-angled` is a deliberate CLI override; with `milestone.strict: true`, YAML `allow_angled: true` alone does not unlock
- On hardware use `SafetyGates(real_robot=True)` (tighter angle limits from `config/gates.yaml`) and `--backend real` with measured `hardware.*` in `config/calib.yaml`

```python
# BAD
robot.move_to_pose(p_aim, Rd)
hand.set_angles(HAND_TRIGGER)

# GOOD
gates.check_drill_angles(tilt, azim, d, BOARD_N)
gates.check_workspace(p_aim, "aim")
gates.check_trigger_clearance(tip, HOLE_TARGET, BOARD_N)
```
