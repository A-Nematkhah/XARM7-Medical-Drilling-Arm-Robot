"""
xArm7 + Inspire RH56E2 + wrist RealSense D435i: find a cordless drill, pick it up,
drill a hole in a board, and put the drill back.

    python run_drill_task.py --viewer
    python run_drill_task.py --video outputs/demo.mp4
    python run_drill_task.py --random 3 --eval --no-place
    python run_drill_task.py --viewer --tilt 15 --azim 10 --allow-angled
    python run_drill_task.py --calib config/calib.yaml --gates config/gates.yaml
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "sim"))

ap = argparse.ArgumentParser()
ap.add_argument("--viewer", action="store_true", help="open the interactive viewer + wrist D435i window")
ap.add_argument("--no-wrist-cam", action="store_true",
                help="do not open the live wrist-camera window (viewer still works)")
ap.add_argument("--video", default=None, help="record an mp4 (headless)")
ap.add_argument("--random", type=int, default=None, help="randomise drill pose with this seed")
ap.add_argument("--xy", type=float, nargs=2, default=(0.47, -0.02))
ap.add_argument("--yaw", type=float, default=0.0, help="drill yaw on the table (deg)")
ap.add_argument("--tilt", type=float, default=0.0,
                help="bit tilt from board normal (deg): +up / -down")
ap.add_argument("--azim", type=float, default=0.0,
                help="bit yaw from board normal (deg): + toward +Y / - toward -Y")
ap.add_argument("--allow-angled", action="store_true",
                help="unlock non-zero tilt/azim (default: perpendicular milestone only)")
ap.add_argument("--no-noise", action="store_true", help="ideal depth (no D435i noise model)")
ap.add_argument("--video-size", type=int, nargs=2, default=(960, 540), help="video width height")
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--no-place", action="store_true", help="skip putting the drill back")
ap.add_argument("--real-robot", action="store_true",
                help="use tighter SafetyGates limits intended for hardware")
ap.add_argument("--backend", choices=("sim", "real"), default="sim",
                help="motion/camera backend (real stub until SDK is wired)")
ap.add_argument("--calib", default=None, help="path to calib YAML (default: config/calib.yaml)")
ap.add_argument("--gates", default=None, help="path to gates YAML (default: config/gates.yaml)")
ap.add_argument("--board-target", choices=("marker", "aruco", "offset"), default="marker",
                help="how to pick the pierce point on the detected board face")
ap.add_argument("--aruco-id", type=int, default=0, help="ArUco id when --board-target aruco")
ap.add_argument("--eval", action="store_true", help="append accuracy metrics for this run to a CSV file")
ap.add_argument("--metrics", default=None, help="CSV path for --eval (default: outputs/metrics.csv)")
ap.add_argument("--run-log", default=None, help="JSONL phase log (default: outputs/run.jsonl)")
args = ap.parse_args()
if not args.viewer:
    default_gl = "glfw" if sys.platform.startswith("win") else "egl"
    os.environ.setdefault("MUJOCO_GL", default_gl)

import cv2
import mujoco
import scene
from robot import REST_Q, ViewPlanner  # REST_Q == hw.defaults.REST_Q
from perception import DrillDetector, BoardDetector
from safety import SafetyGates, GateError
from calibration import hole_from_prior, load_calib
from hw import ArmDriver, make_sim_backend, make_real_backend
from hw.defaults import REST_Q as HW_REST_Q
from task.constants import METRICS_FIELDS
from task.context import TaskCtx
from task.geom import bit_direction
from task.log import RunLog
from task.metrics_io import ensure_metrics_schema
from task.runtime import attach_sim_callbacks
from task.viz import attach_video, attach_viewer, attach_wrist_preview
from task.phases import perceive_and_grasp, locate_board, approach_board, drill_hole, place_drill

calib = load_calib(args.calib)
ALLOW_ANGLED = bool(args.allow_angled or calib.angled_unlocked())
BOARD_N_HINT = calib.board.normal.copy()
HOLE_HINT = hole_from_prior(calib.board)

gates = SafetyGates.from_file(args.gates, real_robot=args.real_robot)
gates.CHUCK_BEHIND_TIP = float(calib.tool.chuck_behind_tip)
gates.CHUCK_R = float(calib.tool.chuck_r)
try:
    gates.check_drill_angles(args.tilt, args.azim, bit_direction(args.tilt, args.azim, BOARD_N_HINT),
                             BOARD_N_HINT, allow_angled=ALLOW_ANGLED)
except GateError as e:
    print(f"FAILED: {e}", file=sys.stderr)
    sys.exit(1)

metrics = {k: "" for k in METRICS_FIELDS}
metrics.update(
    timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    success=0,
    seed="" if args.random is None else int(args.random),
    tilt_deg=float(args.tilt),
    azim_deg=float(args.azim),
    noise=0 if args.no_noise else 1,
    drill_xy_x=float(args.xy[0]),
    drill_xy_y=float(args.xy[1]),
    drill_yaw_deg=float(args.yaw),
)

rng = np.random.default_rng(args.random) if args.random is not None else None
xy = np.array(args.xy) + (rng.uniform(-0.04, 0.04, 2) if rng is not None else 0)
yaw = np.deg2rad(args.yaw + (rng.uniform(-25, 25) if rng is not None else 0))

if args.backend == "real":
    # Live loop talks only to RealArm / RealHand / RealCamera. No MuJoCo scene,
    # no sim callbacks, no viewer. SDK stubs raise until wired.
    model = data = None
    sim_t = time.perf_counter
    backend = make_real_backend(
        hand_eye=calib.hand_eye, force_scale=calib.hand.force_scale,
        min_z=float(calib.camera.min_z))
    print("backend=real: ArmDriver is RealArm (controller API). "
          "SDK not wired — first motion/capture will raise RealBackendNotReady.",
          flush=True)
else:
    model, _ = scene.build_model(drill_xy=tuple(xy), drill_yaw=yaw)
    data = mujoco.MjData(model)
    data.qpos[:7] = REST_Q
    mujoco.mj_forward(model, data)
    sim_t = lambda: data.time
    backend = make_sim_backend(
        model, data, noise=not args.no_noise, seed=args.random or 0,
        hand_eye=calib.hand_eye, force_scale=calib.hand.force_scale,
        camera_calib=calib.camera)
robot = backend.arm
if not isinstance(robot, ArmDriver):
    raise TypeError(f"backend.arm must implement ArmDriver, got {type(robot)}")
if args.backend == "sim":
    robot.q_cmd = REST_Q.copy()
assert np.allclose(robot.rest_q, HW_REST_Q)

st = dict(motor=False, depth=0.0, phase="start", overlay=None, overlay_until=-1, warned=False,
          holding=False,
          board_n=BOARD_N_HINT.copy(), hole=HOLE_HINT.copy(),
          drill_dir=bit_direction(args.tilt, args.azim, BOARD_N_HINT))

run_log_path = args.run_log or os.path.join(HERE, "outputs", "run.jsonl")
runlog = RunLog(run_log_path, sim_t=sim_t, run_meta=dict(
    backend=args.backend, calib=calib.source, gates=gates.source,
    board_target=args.board_target, angled=ALLOW_ANGLED,
    tilt=args.tilt, azim=args.azim, seed=args.random,
))

if model is not None:
    trig_q = int(model.jnt_qposadr[model.joint("trigger").id])
    tip_site = int(model.site("bit_tip").id)
else:
    trig_q, tip_site = -1, -1

ctx = TaskCtx(
    args=args, calib=calib, model=model, data=data, robot=robot, hand=backend.hand,
    cam=backend.camera, views=ViewPlanner(calib.hand_eye, model=model),
    det=DrillDetector(backend.camera, bit_reach=calib.tool.bit_reach,
                      housing_to_tip=calib.tool.housing_to_tip),
    board_det=BoardDetector(backend.camera, hole_offset_yz=tuple(calib.board.hole_yz),
                            target_mode=args.board_target, aruco_id=args.aruco_id),
    gates=gates, st=st, metrics=metrics, runlog=runlog, here=HERE,
    board_n_hint=BOARD_N_HINT, hole_hint=HOLE_HINT, allow_angled=ALLOW_ANGLED,
    sim_t=sim_t, xy=xy, yaw=yaw,
    trig_q=trig_q, tip_site=tip_site,
)
if args.backend == "sim":
    backend.hand.set_angles(backend.hand.OPEN)
    backend.hand.cmd[:] = backend.hand.target
print(f"calib: {calib.source} | gates: {gates.source} | backend={args.backend} | "
      f"board-target={args.board_target} | angled={'unlocked' if ALLOW_ANGLED else 'locked'}",
      flush=True)

if args.backend == "sim":
    attach_sim_callbacks(ctx)
    recorder = attach_video(ctx, args.video, args.video_size, args.fps) if args.video else None
    viewer = attach_viewer(ctx) if args.viewer else None
    if args.viewer and not args.no_wrist_cam:
        attach_wrist_preview(ctx)
else:
    recorder = None
    viewer = None
    if args.viewer or args.video:
        print("viewer/video are sim-only; ignoring with --backend real", flush=True)


def write_metrics(ok, fail_reason=""):
    if not args.eval:
        return
    path = args.metrics or os.path.join(HERE, "outputs", "metrics.csv")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    bak = ensure_metrics_schema(path, METRICS_FIELDS)
    if bak:
        print(f"metrics schema changed; archived {bak}", flush=True)
    metrics["success"] = 1 if ok else 0
    metrics["fail_reason"] = fail_reason
    tnow = float(data.time) if data is not None else float(sim_t())
    metrics["sim_time_s"] = round(tnow, 2)
    metrics["drill_xy_x"] = round(float(xy[0]), 4)
    metrics["drill_xy_y"] = round(float(xy[1]), 4)
    metrics["drill_yaw_deg"] = round(float(np.degrees(yaw)), 2)
    new_file = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=METRICS_FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
        w.writerow({k: metrics.get(k, "") for k in METRICS_FIELDS})
    print(f"metrics appended to {path}", flush=True)


def main():
    perceive_and_grasp(ctx)
    locate_board(ctx)
    approach_board(ctx)
    drill_hole(ctx)
    place_drill(ctx)
    ctx.phase("done")
    robot.wait(1.0)
    ctx.phase_end("done")


if __name__ == "__main__":
    ok = True
    fail_reason = ""
    try:
        main()
    except (RuntimeError, GateError) as e:
        ok = False
        fail_reason = str(e)
        st["holding"] = False
        st["phase"] = f"FAILED: {e}"
        ctx.log(f"FAILED: {e}")
        runlog.event("run", "fail", reason=str(e), failed_phase=st["phase"])
        robot.wait(1.0)
    finally:
        write_metrics(ok, fail_reason)
        runlog.close(success=ok, fail_reason=fail_reason)
        if recorder:
            recorder.close(); print(f"video written to {args.video}")
        os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
        cv2.imwrite(os.path.join(HERE, "outputs", "last_detection.png"),
                    st["overlay"][:, :, ::-1] if st["overlay"] is not None else np.zeros((10, 10, 3), np.uint8))
        print(f"run log: {run_log_path}", flush=True)
        cv2.destroyAllWindows()
    if viewer is not None:
        while viewer.is_running():
            time.sleep(0.05); viewer.sync()
    sys.exit(0 if ok else 1)
