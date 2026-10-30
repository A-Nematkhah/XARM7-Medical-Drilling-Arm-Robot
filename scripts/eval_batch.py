"""Run several randomised sim trials and print a pass/fail summary.

Each seed jitters the face mark and the drill XY left of the fixed board (`--random`).
Pass `--fixed-spawn` after `--` for the default drill/mark.

    python scripts/eval_batch.py --seeds 0-9 --tilt 15 --azim 10
    python scripts/eval_batch.py --seeds 1,2,5 --no-place --min-success 0.8
    python scripts/eval_batch.py --seeds 0-9 -- --fixed-board
"""
import argparse
import csv
import statistics as stats
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "run_drill_task.py"
DEFAULT_METRICS = ROOT / "outputs" / "eval_batch.csv"


def parse_seeds(text):
    seeds = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("no seeds parsed")
    return seeds


def report(rows):
    ok = [r for r in rows if r.get("success") == "1"]
    fail = [r for r in rows if r.get("success") != "1"]
    n = len(rows)
    rate = len(ok) / max(n, 1)
    print(f"batch: {len(ok)}/{n} success ({100 * rate:.0f}%)")
    if fail:
        print("failures:")
        for r in fail:
            print(f"  seed={r.get('seed')}: {r.get('fail_reason')}")

    def mean_sd(key, unit=""):
        xs = [float(r[key]) for r in ok if r.get(key) not in ("", None)]
        if not xs:
            return
        sd = stats.pstdev(xs) if len(xs) > 1 else 0.0
        print(f"  {key:22s} {stats.mean(xs):7.2f} ± {sd:5.2f} {unit}")

    if ok:
        print("successful runs (mean ± std):")
        mean_sd("entry_err_vs_gt_mm", "mm")
        mean_sd("entry_err_vs_cmd_mm", "mm")
        mean_sd("depth_mm", "mm")
        mean_sd("depth_err_mm", "mm")
        mean_sd("depth_gt_mm", "mm")
        mean_sd("approach_lat_mm", "mm")
        mean_sd("approach_clear_mm", "mm")
        mean_sd("bit_angle_err_deg", "deg")
        mean_sd("board_hole_err_mm", "mm")
        mean_sd("cut_force_n", "N")
        mean_sd("cut_torque_nm", "N*m")
        mean_sd("chuck_rpm", "rpm")
    return rate


def main():
    ap = argparse.ArgumentParser(description="Batch-eval run_drill_task.py over several seeds")
    ap.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 1,2,5")
    ap.add_argument("--tilt", type=float, default=0.0)
    ap.add_argument("--azim", type=float, default=0.0)
    ap.add_argument("--allow-angled", action="store_true",
                    help="forward --allow-angled (needed if tilt/azim != 0)")
    ap.add_argument("--metrics", default=str(DEFAULT_METRICS),
                    help="CSV path (default: outputs/eval_batch.csv, not the hand-run metrics.csv)")
    ap.add_argument("--fresh", action="store_true", default=True,
                    help="delete the metrics file before the batch (default)")
    ap.add_argument("--append", action="store_true",
                    help="keep existing rows in --metrics instead of starting fresh")
    ap.add_argument("--min-success", type=float, default=None,
                    help="exit 1 if success rate is below this (0-1)")
    ap.add_argument("--no-place", action="store_true", default=True,
                    help="skip place-back (default on for faster eval)")
    ap.add_argument("--place", action="store_true", help="include place-back phase")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="extra args forwarded to run_drill_task.py (put after --)")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    metrics_path = Path(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and not args.append and metrics_path.exists():
        metrics_path.unlink()

    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]

    rc_fail = 0
    for seed in seeds:
        cmd = [
            args.python, str(TASK),
            "--eval", "--metrics", str(metrics_path),
            "--random", str(seed),
            "--tilt", str(args.tilt),
            "--azim", str(args.azim),
        ]
        if args.no_place and not args.place:
            cmd.append("--no-place")
        if args.allow_angled or abs(args.tilt) > 1e-9 or abs(args.azim) > 1e-9:
            cmd.append("--allow-angled")
        cmd.extend(extra)
        print(f"\n=== seed {seed} ===", flush=True)
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            rc_fail += 1

    if not metrics_path.exists():
        print("no metrics file written", file=sys.stderr)
        sys.exit(1)

    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    print()
    print(f"task exit nonzero: {rc_fail}/{len(seeds)}")
    rate = report(rows)
    if args.min_success is not None and rate + 1e-9 < args.min_success:
        print(f"FAIL: success rate {rate:.0%} < --min-success {args.min_success:.0%}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
