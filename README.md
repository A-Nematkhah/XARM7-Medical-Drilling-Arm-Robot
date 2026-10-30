# xArm7 single-camera drill task

MuJoCo simulation of an xArm7 + Inspire RH56E2 + wrist RealSense D435i that finds a cordless drill, picks it up, drills a hole, and puts the drill back. A 360° LiDAR on the hand (`lidar_eye` in `config/calib.yaml`) coarsely locates the drill and board so the wrist camera can look toward them. Pierce and tip-to-face still close on the wrist D435i. With `--viewer`, one window shows the sim scene with LiDAR (BEV + range) and D435i insets — same layout as `--video`. The same task code can talk to hardware through `hw/` once SDKs and `config/calib.yaml` are filled in.

Safety checks live in `sim/safety.py`. Do not send a planned motion, trigger press, or drill feed to hardware unless it passed those gates (`SafetyGates(real_robot=True)` on a real robot).

## Setup

Python **3.10+** (3.12 preferred) with MuJoCo 3.2+. Create a venv in this repo — do not reuse a venv copied from another machine/user:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On the robot PC only, also install hardware SDKs:

```bat
pip install -r requirements-hw.txt
pip install git+https://github.com/xArm-Developer/xArm-Python-SDK.git
```

Headless OpenGL backend is chosen automatically (`glfw` on Windows, `egl` elsewhere). Override if needed:

```bat
set MUJOCO_GL=glfw
```

## Run the task (sim)

```bat
python run_drill_task.py --viewer
python run_drill_task.py --video outputs/demo.mp4
python run_drill_task.py --random 3 --eval --no-place
python run_drill_task.py --fixed-spawn --eval --no-place
python run_drill_task.py --real-robot --calib config/calib.yaml --gates config/gates.yaml
```

Each sim run **randomizes the face mark and the drill** (board pose stays fixed). Drill XY stays in the left-of-board patch (x 0.30–0.58 m, y −0.14–0.28 m; cyan box). Pass `--random SEED` to reproduce; `--fixed-spawn` keeps the default drill/mark. Aim stays on `BoardDetector`; spawn pose is eval GT only.

`--real-robot` only tightens software gates (use while still on `--backend sim` to rehearse hardware limits). It does **not** open the xArm controller by itself.

Each run writes `outputs/run.jsonl` (override with `--run-log`) with one JSON object per phase start/end, plus tracking/collision fields.

## Calib and perpendicular dry-check

1. Copy `config/calib.example.yaml` → `config/calib.yaml` if needed.
2. On hardware, replace **measured** `hand_eye`, `board`, `tool`, `hand.force_scale`, and the `hardware:` block (`arm_ip`, `hand_port`, optional `rs_serial`).
3. Validate:

```bat
python scripts/check_calib.py
python scripts/check_calib.py --real-robot
python scripts/check_calib.py --require-hardware
```

Board pose in calib is a **search prior** only. After grasp, aiming uses `BoardDetector`.

## Hardware backend

```bat
python run_drill_task.py --backend real --real-robot --calib config/calib.yaml
```

`--backend real` uses `hw.real_backend` (`RealArm` / `RealHand` / `RealCamera`):

| Device | API |
|--------|-----|
| xArm7 | `XArmAPI` — `set_position` / `set_servo_angle` / controller IK |
| RH56 | Modbus-RTU over `hardware.hand_port` (pyserial) |
| D435i | `pyrealsense2` aligned RGB-D |

Live motion must stay on the controller — never port MuJoCo `servo_step` / `mj_jacSite` into the hardware loop. Without SDKs or with placeholder IPs, calls raise `RealBackendNotReady`.

## Tests, batch eval, angled unlock

Full bug-hunt suite (pytest + multi-scenario task runs, grouped fail reasons):

```bat
python scripts/eval_suite.py --preset bugs
python scripts/eval_suite.py --preset quick
python scripts/eval_suite.py --preset full --min-success 0.8
python scripts/eval_suite.py --list
```

Writes `outputs/eval_suite_report.txt`, `outputs/eval_suite_results.json`, and `outputs/eval_suite.csv`. Use `--preset bugs` after a tracking/clearance fail (e.g. approach-far) to stress relocate, board-target, and random seeds.

Keep angled holes **locked** until perpendicular batch eval is stable:

```bat
pytest
python scripts/eval_batch.py --seeds 0-9 --min-success 0.8
python scripts/unlock_angled.py
python scripts/unlock_angled.py --dry-run
```

`scripts/unlock_angled.py` reads `outputs/eval_batch.csv` (or `milestone.baseline_metrics`), checks `min_perp_success` / `min_perp_runs` from calib, then writes `config/angled_unlocked`. With `milestone.strict: true` (default), setting `allow_angled: true` in YAML alone does **not** unlock.

A checked-in sim baseline lives in `outputs/baselines/perp_seeds_0-9.csv` (10/10 success). Local `outputs/eval_*.csv` and `config/angled_unlocked` remain gitignored.

After unlock (or with an explicit CLI override):

```bat
python run_drill_task.py --viewer --tilt 15 --azim 10
python run_drill_task.py --viewer --tilt 15 --azim 10 --allow-angled
python scripts/eval_batch.py --seeds 0-9 --tilt 15 --azim 10 --allow-angled --min-success 0.8
```

`--allow-angled` is a deliberate developer override; prefer the unlock script for the normal path.

Summarize a metrics CSV:

```bat
python outputs/summarize_metrics.py
```

## Layout

| Path | Role |
|------|------|
| `run_drill_task.py` | CLI + world setup; calls `task.phases` |
| `task/phases/` | grasp / board / approach / drill / place |
| `task/log.py` | JSONL phase log (`outputs/run.jsonl`) |
| `hw/` | Arm / hand / camera backends (sim + real) |
| `sim/safety.py` | Safety gates; limits from `config/gates.yaml` |
| `sim/calibration.py` | Load `config/calib.yaml` |
| `sim/perception.py` | Wrist RGB-D drill + board / marker detectors |
| `sim/lidar.py` | Wrist 360° LiDAR; coarse drill/board XY for D435i view planning |
| `sim/robot.py` | Sim IK / servo / Inspire hand |
| `sim/scene.py` | MuJoCo scene (board target pad+dot, wrist LiDAR puck) |
| `config/` | `calib.yaml`, `gates.yaml`, `motion.yaml`, optional `angled_unlocked` |
| `scripts/eval_batch.py` | Multi-seed eval → `outputs/eval_batch.csv` |
| `scripts/unlock_angled.py` | Unlock angled after perp baseline |
| `scripts/check_calib.py` | Calib / gate sanity check |
| `AGENTS.md` | Hardware transfer / safety-gate constraints |

Live aim uses TCP + `tip_in_hand` (perception at grasp, refreshed from the wrist D435i). Sim may fall back to the `bit_tip` site only if that wrist observation is empty — hardware has no such site. Motor ON follows the RH56 index command. Lift/depth gates use TCP height and estimated tip-to-plane clearance. GT fields (`entry_err_vs_gt_mm`, `depth_gt_mm`) are report-only.

`cam.pose()` is TCP FK × `hand_eye` from `config/calib.yaml` (not MuJoCo `cam_xpos`). Wrist LiDAR origin is TCP FK × `lidar_eye`. Finger load is `get_forces() * hand.force_scale`. Grasp closes until wrap-finger load exceeds `gates.finger_load` (`task/grasp_ctrl.py`) and freezes those commands so squeeze holds; `check_grasp` still aborts if too few fingers loaded. Board collision is gated geometrically (detected plane + hand/housing samples) even when MuJoCo `ncon` is absent. Bit length is `tool.bit_reach`.

Phase 4: live motion goes through `ArmDriver`. Sim `move_to_pose` is min-jerk + Cartesian PI servo (`sim/control.py`); `move_along` is the tool-axis feed/retract primitive; `move_joint_path` is one min-jerk through the IK chain (no stop at each via). Per-phase TCP speed caps live in `config/motion.yaml` (`scan` / `grasp` / `lift` / `relocate` / `approach` / `feed` / `retract` / `place`); feed is also clamped by `gates.max_feed_speed`. Drill feed is a depth PI (`task/feed_ctrl.py`): `v ∝ (target − pen)`, slower in the last millimetres, and further derated when cutting thrust is high or chuck RPM sags (`feed.f_nom` / `f_hi` / `rpm_nom` / `rpm_lo`). Missing tach (`chuck_rpm is None`) leaves the RPM scale at 1; `rpm=0` with the motor on is a stall. On `--backend real` set `hand.force_scale` so RH56 `FORCE_ACT` (0..1000) maps into N·m (e.g. `0.001`); sim keeps `1.0`. Grasp closes until wrap-finger load exceeds `gates.finger_load` (`task/grasp_ctrl.py`). IK branches (`plan_path_ik` / `plan_ik` / `ik_solutions`) pick the lowest `hw/ik_select.branch_cost` (short joint path, little ±2π wrap, near rest) among chains that still clear `min_margin` — not the max-margin pose if it flips the elbow. The hardware path must not call `mj_jacSite`. Joint-limit gates use `robot.get_q()`.

Board pierce / normal from the wrist detector are EMA-filtered (`task/target_filter.py`): jumps above `MAX_HOLE_JUMP` are ignored; accepted updates blend with `MARKER_ALPHA` / `MARKER_N_ALPHA` and re-project onto the face plane during approach.

Phase 5: chuck↔board contact is real (bit geoms stay non-colliding so the flute can occupy the hole). While the motor is on and the tip is in the face, `sim/cutting.py` applies axial thrust and chuck torque (`config/calib.yaml` `cutting:`). Pushing the tip in without spin aborts (`check_dry_cut`); a stalled chuck aborts (`check_cut_spin`). Contact force above `max_board_contact_n` aborts. Wrist RGB-D can apply `camera.latency_s` / edge noise from calib. GT fields (`entry_err_vs_gt_mm`, `depth_gt_mm`) are report-only — never gates.
