"""Task-level constants that are not safety-gate limits."""
import numpy as np

UP = np.array([0.0, 0.0, 1.0])
GRIP_BELOW_HOUSING = 0.051
SLIDE = 0.16
STANDOFF = 0.008
# Index 0..1000: 0 = closed (trigger pressed), 1000 = extended (off). The finger
# usually stops around ~300 once it is against the trigger, so ON is above that.
INDEX_TRIGGER_ON = 400
INDEX_TRIGGER_OFF = 700
MOTOR_SPEED = 150.0
AIM_STANDOFF = 0.03
APPROACH_CLEAR = 0.14
CLEAR_TOL = 0.003
LAT_TOL = 0.002               # planar (Y/Z) settle tolerance (m)
MAX_APPROACH_STEP = 0.03      # max TCP step per visual-servo tick (m)
MAX_HOLE_JUMP = 0.05          # reject marker updates that teleport (m)
DRILL_DEPTH = 0.025
FEED_MARGIN = 0.003              # max overshoot budget past DRILL_DEPTH (was 8 mm open-loop fudge)
DEPTH_TOL = 0.001                # stop feed when measured depth within this of target
MAX_FEED_STEP = 0.004            # closed-loop chunk along the bit (m)
HAND_GRASP = [0, 0, 0, 1000, 0, 0]
HAND_TRIGGER = [0, 0, 0, 0, 0, 0]
SAFE_TIP = 0.08

METRICS_FIELDS = [
    "timestamp", "success", "fail_reason", "seed", "tilt_deg", "azim_deg", "noise",
    "drill_xy_x", "drill_xy_y", "drill_yaw_deg",
    "coarse_xy_err_mm", "grip_err_mm", "drill_yaw_err_deg",
    "board_normal_err_deg", "board_hole_err_mm", "target_source",
    "approach_clear_mm", "approach_lat_mm",
    "entry_err_vs_cmd_mm", "entry_err_vs_gt_mm", "entry_err_vs_marker_mm",
    "entry_err_y_mm", "entry_err_z_mm",
    "bit_angle_err_deg", "depth_mm", "depth_err_mm", "depth_gt_mm",
    "place_disp_mm", "sim_time_s",
]
