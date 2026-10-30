"""
xArm7 + Inspire RH56E2 + wrist RealSense D435i: find a cordless drill, pick it up,
drill a hole in a board, and put the drill back.

    python run_drill_task.py --viewer
    python run_drill_task.py --video outputs/demo.mp4
    python run_drill_task.py --random 3 --eval --no-place
    python run_drill_task.py --fixed-spawn --eval --no-place
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
ap.add_argument("--viewer", action="store_true",
                help="open one window: sim scene + wrist D435i + LiDAR insets")
ap.add_argument("--no-wrist-cam", action="store_true",
                help="hide the wrist-camera inset in the viewer")
ap.add_argument("--no-lidar-view", action="store_true",
                help="hide the LiDAR inset and skip live scan refresh")
ap.add_argument("--video", default=None, help="record an mp4 (headless)")
ap.add_argument("--random", type=int, default=None,
                help="spawn seed (sim default: fresh seed each run). "
                     "Fixed board; random face mark + left-of-board drill")
ap.add_argument("--fixed-spawn", action="store_true",
                help="use default drill XY/yaw and face mark (no randomization)")
ap.add_argument("--fixed-board", action="store_true",
                help="alias of --fixed-spawn (board pose is always fixed anyway)")
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
ap.add_argument("--motion", default=None, help="path to motion profiles YAML (default: config/motion.yaml)")
ap.add_argument("--board-target", choices=("marker", "aruco", "offset"), default="marker",
                help="how to pick the pierce point on the detected board face")
ap.add_argument("--aruco-id", type=int, default=0, help="ArUco id when --board-target aruco")
ap.add_argument("--eval", action="store_true", help="append accuracy metrics for this run to a CSV file")
ap.add_argument("--metrics", default=None, help="CSV path for --eval (default: outputs/metrics.csv)")
ap.add_argument("--run-log", default=None, help="JSONL phase log (default: outputs/run.jsonl)")
args = ap.parse_args()
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
from task.motion_profiles import load_motion_profiles
from task.runtime import attach_hw_callbacks, attach_sim_callbacks
from task.state_machine import TaskStateMachine
from task.viz import attach_lidar_preview, attach_video, attach_viewer
from task.phases import perceive_and_grasp, locate_board, approach_board, drill_hole, place_drill

calib = load_calib(args.calib)
ALLOW_ANGLED = bool(args.allow_angled or calib.angled_unlocked())
BOARD_N_HINT = calib.board.normal.copy()
HOLE_HINT = hole_from_prior(calib.board)

real_robot = bool(args.real_robot or args.backend == "real")
gates = SafetyGates.from_file(args.gates, real_robot=real_robot)
gates.CHUCK_BEHIND_TIP = float(calib.tool.chuck_behind_tip)
gates.CHUCK_R = float(calib.tool.chuck_r)
motion = load_motion_profiles(args.motion, feed_vmax_cap=gates.MAX_FEED_SPEED)
try:
    gates.check_drill_angles(args.tilt, args.azim, bit_direction(args.tilt, args.azim, BOARD_N_HINT),
                             BOARD_N_HINT, allow_angled=ALLOW_ANGLED)
except GateError as e:
    print(f"FAILED: {e}", file=sys.stderr)
    sys.exit(1)

fixed_spawn = bool(args.fixed_spawn or args.fixed_board)
# Every sim run randomizes drill + face mark unless --fixed-spawn / --fixed-board.
if args.backend == "sim" and not fixed_spawn and args.random is None:
    args.random = int(np.random.SeedSequence().generate_state(1)[0] % (2**31 - 1))

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
xy = np.array(args.xy, float)
yaw = np.deg2rad(args.yaw)
board_pos = scene.BOARD_POS.copy()
hole_yz = scene.HOLE_YZ.copy()
board_yaw = 0.0
if rng is not None and args.backend == "sim" and not fixed_spawn:
    xy, yaw, board_pos, hole_yz, board_yaw = scene.sample_layout(rng)
    metrics["seed"] = int(args.random)
    metrics["drill_xy_x"] = float(xy[0])
    metrics["drill_xy_y"] = float(xy[1])
    metrics["drill_yaw_deg"] = float(np.degrees(yaw))
hole_gt, board_n_gt = scene.hole_on_face(board_pos, hole_yz, board_yaw)

if args.backend == "real":
    # Live loop talks only to RealArm / RealHand / RealCamera. No MuJoCo scene,
    # no sim callbacks, no viewer. Missing SDK / devices -> RealBackendNotReady.
    model = data = None
    sim_t = time.perf_counter
    hw = calib.hardware
    backend = make_real_backend(
        arm_ip=hw.arm_ip, hand_port=hw.hand_port or None,
        hand_eye=calib.hand_eye, force_scale=calib.hand.force_scale,
        min_z=float(calib.camera.min_z), rs_serial=hw.rs_serial or None,
        hand_baud=hw.hand_baud, hand_slave_id=hw.hand_slave_id,
        tcp_speed_mm_s=hw.tcp_speed_mm_s, joint_speed_deg_s=hw.joint_speed_deg_s,
        connect=bool(hw.connect_on_start),
        lidar_eye=calib.lidar_eye)
    print("backend=real: ArmDriver is RealArm (controller API, no mj_jacSite). "
          f"arm_ip={hw.arm_ip!r} hand_port={hw.hand_port!r}. "
          "Install requirements-hw.txt; first motion/capture connects lazily "
          "unless hardware.connect_on_start is true.",
          flush=True)
else:
    model, _ = scene.build_model(
        drill_xy=tuple(xy), drill_yaw=yaw,
        board_pos=board_pos, hole_yz=hole_yz, board_yaw=board_yaw)
    data = mujoco.MjData(model)
    data.qpos[:7] = REST_Q
    mujoco.mj_forward(model, data)
    sim_t = lambda: data.time
    backend = make_sim_backend(
        model, data, noise=not args.no_noise, seed=args.random or 0,
        hand_eye=calib.hand_eye, force_scale=calib.hand.force_scale,
        camera_calib=calib.camera, lidar_eye=calib.lidar_eye)
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
    board_pos=np.round(board_pos, 4).tolist(),
    hole_yz=np.round(hole_yz, 4).tolist(),
    board_yaw_deg=round(float(np.degrees(board_yaw)), 2),
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
    lidar=getattr(backend, "lidar", None),
    gates=gates, motion=motion, st=st, metrics=metrics, runlog=runlog, here=HERE,
    board_n_hint=BOARD_N_HINT, hole_hint=HOLE_HINT, allow_angled=ALLOW_ANGLED,
    sim_t=sim_t, xy=xy, yaw=yaw,
    trig_q=trig_q, tip_site=tip_site,
    hole_gt=hole_gt if args.backend == "sim" else None,
    board_n_gt=board_n_gt if args.backend == "sim" else None,
    state_machine=TaskStateMachine(),
)
if args.backend == "sim":
    backend.hand.set_angles(backend.hand.OPEN)
    backend.hand.cmd[:] = backend.hand.target
print(f"calib: {calib.source} | gates: {gates.source} | motion: {motion.source} | "
      f"backend={args.backend} | real_robot={real_robot} | "
      f"board-target={args.board_target} | "
      f"angled={'unlocked' if ALLOW_ANGLED else 'locked'}",
      flush=True)
if args.backend == "sim":
    seed_s = "fixed" if fixed_spawn else f"seed {args.random}"
    print(f"spawn ({seed_s}): drill ({xy[0]:.3f}, {xy[1]:.3f}) "
          f"yaw {np.degrees(yaw):+.1f} deg | "
          f"board ({board_pos[0]:.3f}, {board_pos[1]:.3f}, {board_pos[2]:.3f}) "
          f"mark yz ({hole_yz[0]:+.3f}, {hole_yz[1]:+.3f}) "
          f"(GT for eval only; aim from BoardDetector)",
          flush=True)

if args.backend == "sim":
    attach_sim_callbacks(ctx)
    recorder = attach_video(ctx, args.video, args.video_size, args.fps) if args.video else None
    viewer = None
    if args.viewer:
        viewer = attach_viewer(ctx, wrist=not args.no_wrist_cam, lidar=not args.no_lidar_view,
                               size=tuple(args.video_size))
    if (args.viewer or args.video) and not args.no_lidar_view:
        attach_lidar_preview(ctx, window=False)
else:
    attach_hw_callbacks(ctx)
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
    metrics["board_pos_x"] = round(float(board_pos[0]), 4)
    metrics["board_pos_y"] = round(float(board_pos[1]), 4)
    metrics["board_pos_z"] = round(float(board_pos[2]), 4)
    metrics["board_yaw_deg"] = round(float(np.degrees(board_yaw)), 2)
    metrics["hole_yz_y"] = round(float(hole_yz[0]), 4)
    metrics["hole_yz_z"] = round(float(hole_yz[1]), 4)
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
    except Exception as e:
        ok = False
        fail_reason = str(e)
        st["holding"] = False
        st["phase"] = f"FAILED: {e}"
        ctx.log(f"FAILED: {e}")
        runlog.event("run", "fail", reason=str(e), failed_phase=st["phase"])
        try:
            robot.wait(1.0)
        except Exception:
            pass
        if not isinstance(e, (RuntimeError, GateError)):
            raise
    finally:
        write_metrics(ok, fail_reason)
        runlog.close(success=ok, fail_reason=fail_reason)
        if recorder:
            recorder.close(); print(f"video written to {args.video}")
        os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
        cv2.imwrite(os.path.join(HERE, "outputs", "last_detection.png"),
                    st["overlay"][:, :, ::-1] if st["overlay"] is not None else np.zeros((10, 10, 3), np.uint8))
        print(f"run log: {run_log_path}", flush=True)
        if viewer is None:
            cv2.destroyAllWindows()
        else:
            print("close the Smooty window to exit", flush=True)
            while True:
                try:
                    if cv2.getWindowProperty(viewer, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
                cv2.waitKey(50)
            cv2.destroyAllWindows()
    sys.exit(0 if ok else 1)
