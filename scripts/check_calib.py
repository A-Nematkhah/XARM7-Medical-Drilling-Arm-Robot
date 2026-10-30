"""Validate calib.yaml for sim gates and (optionally) real hardware wiring.

    python scripts/check_calib.py
    python scripts/check_calib.py --real-robot
    python scripts/check_calib.py --require-hardware
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))

from calibration import hole_from_prior, load_calib  # noqa: E402
from safety import SafetyGates  # noqa: E402
from task.geom import bit_direction  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=None)
    ap.add_argument("--gates", default=None)
    ap.add_argument("--real-robot", action="store_true",
                    help="load tighter SafetyGates limits")
    ap.add_argument("--require-hardware", action="store_true",
                    help="fail if arm_ip / hand_port still placeholders")
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--azim", type=float, default=0.0)
    args = ap.parse_args()

    cfg = load_calib(args.calib)
    gates = SafetyGates.from_file(args.gates, real_robot=args.real_robot)
    print(f"calib: {cfg.source}")
    print(f"gates: {gates.source}  real_robot={args.real_robot}")

    n = cfg.board.normal
    hole = hole_from_prior(cfg.board)
    print(f"board prior centre {np.round(cfg.board.position, 4).tolist()}")
    print(f"board normal       {np.round(n, 4).tolist()}")
    print(f"hole prior         {np.round(hole, 4).tolist()}")
    print(f"hand_eye t         {np.round(cfg.hand_eye.t, 4).tolist()}")
    print(f"hand_eye rpy_deg   {np.round(cfg.hand_eye.rpy_deg, 2).tolist()}")
    print(f"lidar_eye t        {np.round(cfg.lidar_eye.t, 4).tolist()}")
    print(f"lidar_eye rpy_deg  {np.round(cfg.lidar_eye.rpy_deg, 2).tolist()}")
    print(f"tool held_bit_reach {cfg.tool.held_bit_reach:.4f} m")
    print(f"hand force_scale    {cfg.hand.force_scale}")
    if (args.real_robot or args.require_hardware) and float(cfg.hand.force_scale) >= 0.1:
        print(
            "FAIL: hand.force_scale looks like sim N*m (RH56 FORCE_ACT is 0..1000). "
            "Set force_scale so get_forces()*scale is in N*m (e.g. 0.001).",
            file=sys.stderr)
        sys.exit(1)

    d = bit_direction(args.tilt, args.azim, n)
    allow = cfg.angled_unlocked(cli_allow=False)
    print(f"angled unlocked     {allow} (strict={cfg.milestone.strict})")
    try:
        gates.check_drill_angles(args.tilt, args.azim, d, n, allow_angled=allow)
        print(f"drill angles        OK (tilt={args.tilt}, azim={args.azim})")
    except Exception as e:
        print(f"drill angles        FAIL: {e}")
        if abs(args.tilt) > 1e-9 or abs(args.azim) > 1e-9:
            sys.exit(1)

    hw = cfg.hardware
    print(f"hardware.arm_ip     {hw.arm_ip!r}")
    print(f"hardware.hand_port  {hw.hand_port!r}")
    print(f"hardware.rs_serial  {hw.rs_serial!r}")
    placeholder = ("xxx" in hw.arm_ip.lower()) or (not hw.hand_port.strip())
    if args.require_hardware and placeholder:
        print("FAIL: set hardware.arm_ip and hardware.hand_port for --backend real",
              file=sys.stderr)
        sys.exit(1)
    if placeholder:
        print("note: hardware block still has placeholders (OK for sim)")
    else:
        print("hardware block looks filled (still need SDK + live devices)")

    print("check_calib: OK")


if __name__ == "__main__":
    main()
