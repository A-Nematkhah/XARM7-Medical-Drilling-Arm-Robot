# xArm7 single-camera drill task

MuJoCo simulation of an xArm7 + Inspire RH56E2 + wrist RealSense D435i that finds a cordless drill, picks it up, drills a hole, and puts the drill back.

Safety checks live in `sim/safety.py`. Do not send a planned motion, trigger press, or drill feed to hardware unless it passed those gates (`SafetyGates(real_robot=True)` on a real robot).

## Setup

Python **3.12** is required (MuJoCo 3.2+). Create a venv in this repo — do not reuse a venv copied from another machine/user:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Headless OpenGL backend is chosen automatically (`glfw` on Windows, `egl` elsewhere). Override if needed:

```bat
set MUJOCO_GL=glfw
```

Git is expected on `PATH` (`Git for Windows`). This tree already has a `.git` directory; add a remote when you have a host:

```bat
git remote add origin <url>
git push -u origin master
```

## Run the task

```bat
python run_drill_task.py --viewer
python run_drill_task.py --video outputs/demo.mp4
python run_drill_task.py --random 3 --eval --no-place
python run_drill_task.py --viewer --tilt 15 --azim 10 --allow-angled
python run_drill_task.py --real-robot --calib config/calib.yaml --gates config/gates.yaml
```

`--real-robot` only tightens the software gates; it does not talk to an xArm controller. `--backend real` uses `hw.real_backend.RealArm` (controller API, no `mj_jacSite` / Python `servo_step`); motion/capture raise `RealBackendNotReady` until the SDK is wired. The task only calls `ArmDriver` methods (`move_to_pose`, `move_joints`, `get_q`, …).

Each run writes `outputs/run.jsonl` (override with `--run-log`) with one JSON object per phase start/end, plus tracking/collision fields.

## Before hardware

| Piece | What to use |
|-------|-------------|
| Drivers | `hw/` interfaces; sim via `--backend sim`, real stub via `--backend real` |
| Calib | `config/calib.yaml` (copy from `calib.example.yaml`); hand-eye + board prior |
| Milestone | default is **perpendicular only**; unlock angles with `--allow-angled`, `milestone.allow_angled: true`, or empty file `config/angled_unlocked` |
| Hole target | `--board-target marker` (default, high-contrast disk on the face), or `aruco` / `offset` |

## Tests and batch eval

```bat
pytest
python scripts/eval_batch.py --seeds 0-9 --min-success 0.8
python scripts/eval_batch.py --seeds 0-9 --tilt 15 --azim 10 --allow-angled --min-success 0.8
```

## Layout

| Path | Role |
|------|------|
| `run_drill_task.py` | CLI + world setup; calls `task.phases` |
| `task/phases/` | grasp / board / approach / drill / place |
| `task/log.py` | JSONL phase log (`outputs/run.jsonl`) |
| `hw/` | Arm / hand / camera backends (sim + real stub) |
| `sim/safety.py` | Safety gates; limits from `config/gates.yaml` |
| `sim/calibration.py` | Load `config/calib.yaml` |
| `sim/perception.py` | Wrist RGB-D drill + board / marker detectors |
| `sim/robot.py` | Sim IK / servo / Inspire hand |
| `sim/scene.py` | MuJoCo scene (includes board target pad+dot) |
| `config/` | `calib.yaml`, `gates.yaml`, optional `angled_unlocked` |
| `AGENTS.md` | Hardware transfer / safety-gate constraints |

Board pose in `config/calib.yaml` is a search prior only (`position`, `normal`, `hole_yz`, `half`). After grasp, aiming uses `BoardDetector`. Transit checks the estimated tip against that detected face and the calibrated half-size — not MuJoCo `BOARD_POS`.

The live loop never reads the MuJoCo `bit_tip` site, trigger `qpos`, or drill-body pose. Tip = TCP + `tip_in_hand` (from perception at grasp); motor ON follows the RH56 index command; lift/depth gates use TCP height and estimated tip-to-plane clearance. Sim ground truth is written only into eval fields (`entry_err_vs_gt_mm`, `depth_gt_mm`).

`cam.pose()` is TCP FK × `hand_eye` from `config/calib.yaml` (not MuJoCo `cam_xpos`). Finger load is `get_forces() * hand.force_scale`. Board collision is gated geometrically (detected plane + hand/housing samples) even when MuJoCo `ncon` is absent. Bit length is `tool.bit_reach`.

Phase 4: live motion goes through `ArmDriver`. Sim may implement `move_to_pose` with internal IK; the hardware path (`hw/real_backend.py`) must not call `mj_jacSite`. Joint-limit gates use `robot.get_q()` (measured / controller angles).

Phase 5: chuck↔board contact is real (bit geoms are non-colliding; cutting is kinematic). Pushing the tip into the face without the motor aborts (`check_dry_cut`). Contact force above `max_board_contact_n` aborts. Wrist RGB-D can apply `camera.latency_s` / edge noise from calib. GT fields (`entry_err_vs_gt_mm`, `depth_gt_mm`) are report-only — never gates.
