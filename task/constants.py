"""Task-level constants that are not safety-gate limits."""
import numpy as np

UP = np.array([0.0, 0.0, 1.0])
GRIP_BELOW_HOUSING = 0.051
SLIDE = 0.16
STANDOFF = 0.008
# TCP height while translating to the drill. Pinky/LiDAR sits ~8 cm below TCP
# when grasp Y is world-up; flattening the path to grip Z drives the hand into
# the table. Drop to grip Z only once, beside the drill. Rolled wrists need
# extra height so the puck still clears TABLE_CLEAR_Z.
TRANSIT_TCP_Z = 0.22
# Lift / reorient above this before changing wrist orientation (joint interp
# from the side-view camera pose otherwise dips the hand through the table).
REORIENT_TCP_Z = 0.28
TABLE_CLEAR_Z = 0.015
# Table XY box for top-down search when LiDAR misses (pad around scene.DRILL_SPAWN_*).
# Kept in sync with sim/scene.py spawn (left wrap + hand envelope inside workspace).
DRILL_SCAN_LO = np.array([0.40, -0.18])
DRILL_SCAN_HI = np.array([0.59, 0.15])
# Viewer overlay of the sim spawn patch. Not a live aim target (BoardDetector is).
SPAWN_FACE_XY = np.array([0.64, -0.34])
SPAWN_FACE_EXTENT_XY = np.array([0.012, 0.13 * np.sqrt(1.5)])
SPAWN_DRILL_LO = np.array([0.442267, -0.14])
SPAWN_DRILL_HI = np.array([0.55, 0.103560])
# Index 0..1000: 0 = closed (trigger pressed), 1000 = extended (off). The finger
# often stops ~300–420 against the trigger pad; ON must sit above that contact.
INDEX_TRIGGER_ON = 480
INDEX_TRIGGER_OFF = 700
MOTOR_SPEED = 150.0
AIM_STANDOFF = 0.03
APPROACH_CLEAR = 0.14
CLEAR_TOL = 0.003
LAT_TOL = 0.002               # planar (Y/Z) settle during the long approach (m)
AIM_LAT_TOL = 0.0005          # post-trigger / final aim planar settle (m)
FEED_CTRL_TOL = 0.00015       # DepthFeedCtrl stop band (m); crawl until DRILL_DEPTH
ENTRY_HIT_PEN = 0.0003        # first pierce capture for eval (m)
MAX_APPROACH_STEP = 0.02      # max TCP step per visual-servo tick (m); < AIM_STANDOFF
MAX_HOLE_JUMP = 0.05          # reject marker updates that teleport (m)
MARKER_ALPHA = 0.35           # EMA weight of new pierce-point measurement
MARKER_N_ALPHA = 0.20         # EMA weight of new face-normal measurement
MARKER_MIN_N_DOT = 0.97       # refuse normal updates that diverge / flip
DRILL_DEPTH = 0.025
FEED_MARGIN = 0.003              # max overshoot budget past DRILL_DEPTH (was 8 mm open-loop fudge)
DEPTH_TOL = 0.001                # stop feed when measured depth within this of target
MAX_FEED_STEP = 0.004            # default chunk cap; overridden by motion.yaml feed.max_step
HAND_GRASP = [0, 0, 0, 1000, 0, 0]
# Index closed (trigger). Never send this as a 6-vector — wrap fingers stay
# at grasp_cmd via overlay_index. Kept as a named fallback for the index joint.
HAND_TRIGGER = [0, 0, 0, 0, 0, 0]
SAFE_TIP = 0.08

METRICS_FIELDS = [
    "timestamp", "success", "fail_reason", "seed", "tilt_deg", "azim_deg", "noise",
    "drill_xy_x", "drill_xy_y", "drill_yaw_deg",
    "board_pos_x", "board_pos_y", "board_pos_z", "board_yaw_deg",
    "hole_yz_y", "hole_yz_z",
    "coarse_xy_err_mm", "grip_err_mm", "drill_yaw_err_deg",
    "board_normal_err_deg", "board_hole_err_mm", "target_source", "locate_source",
    "lidar_drill_err_mm", "lidar_board_err_mm",
    "approach_clear_mm", "approach_lat_mm",
    "entry_err_vs_cmd_mm", "entry_err_vs_gt_mm", "entry_err_vs_marker_mm",
    "entry_err_y_mm", "entry_err_z_mm",
    "bit_angle_err_deg", "depth_mm", "depth_err_mm", "depth_gt_mm",
    "cut_force_n", "cut_torque_nm", "chuck_rpm",
    "place_disp_mm", "sim_time_s",
    "grasp_verified", "lift_verified", "board_reloc_ok", "replan_count",
]
