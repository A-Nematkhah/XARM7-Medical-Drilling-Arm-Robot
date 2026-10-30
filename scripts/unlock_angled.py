"""Unlock angled drilling only after a stable perpendicular batch eval.

    python scripts/eval_batch.py --seeds 0-9 --min-success 0.8
    python scripts/unlock_angled.py
    python scripts/unlock_angled.py --metrics outputs/eval_batch.csv --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sim"))

from calibration import load_calib  # noqa: E402


def evaluate_baseline(metrics_path: Path, min_success: float, min_runs: int):
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing metrics CSV: {metrics_path}")
    rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    # Perpendicular-only rows (tilt/azim ~ 0). Missing columns => treat as perp.
    perp = []
    for r in rows:
        tilt = float(r.get("tilt_deg") or 0.0)
        azim = float(r.get("azim_deg") or 0.0)
        if abs(tilt) < 1e-6 and abs(azim) < 1e-6:
            perp.append(r)
    if len(perp) < min_runs:
        raise RuntimeError(
            f"need >= {min_runs} perpendicular rows, found {len(perp)} in {metrics_path}"
        )
    ok = sum(1 for r in perp if r.get("success") == "1")
    rate = ok / len(perp)
    if rate + 1e-9 < min_success:
        raise RuntimeError(
            f"perpendicular success {rate:.0%} ({ok}/{len(perp)}) "
            f"< required {min_success:.0%}"
        )
    return {
        "n": len(perp),
        "ok": ok,
        "success_rate": round(rate, 4),
        "metrics": str(metrics_path.as_posix()),
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main():
    ap = argparse.ArgumentParser(description="Write angled_unlocked after perp baseline passes")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--metrics", default=None, help="override milestone.baseline_metrics")
    ap.add_argument("--min-success", type=float, default=None)
    ap.add_argument("--min-runs", type=int, default=None)
    ap.add_argument("--unlock-file", default=None)
    ap.add_argument("--dry-run", action="store_true", help="check only; do not write unlock file")
    ap.add_argument("--force", action="store_true", help="overwrite an existing unlock file")
    args = ap.parse_args()

    cfg = load_calib(args.calib)
    ms = cfg.milestone
    metrics_path = Path(args.metrics) if args.metrics else cfg.baseline_metrics_path()
    if not metrics_path.is_absolute():
        metrics_path = ROOT / metrics_path
    min_success = float(args.min_success if args.min_success is not None else ms.min_perp_success)
    min_runs = int(args.min_runs if args.min_runs is not None else ms.min_perp_runs)
    unlock = Path(args.unlock_file) if args.unlock_file else cfg.unlock_path()
    if not unlock.is_absolute():
        unlock = ROOT / unlock

    stamp = evaluate_baseline(metrics_path, min_success, min_runs)
    stamp.update(
        min_perp_success=min_success,
        min_perp_runs=min_runs,
        calib=cfg.source,
    )
    print(f"baseline OK: {stamp['ok']}/{stamp['n']} ({100 * stamp['success_rate']:.0f}%)")
    if args.dry_run:
        print(json.dumps(stamp, indent=2))
        return

    if unlock.is_file() and not args.force:
        print(f"already unlocked: {unlock} (pass --force to overwrite)")
        return

    unlock.parent.mkdir(parents=True, exist_ok=True)
    unlock.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {unlock}")
    print("angled drilling unlocked for subsequent runs (no --allow-angled needed)")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
