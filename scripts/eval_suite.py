"""Comprehensive sim eval suite — unit tests + multi-scenario task runs.

Surfaces gate failures (tracking, clearance, board, feed) across spawn modes
so regressions show up as grouped `fail_reason` counts, not one-off viewer runs.

Presets:
  quick   unit tests + a few smokes (default spawn, seed 0/3, fixed-board)
  bugs    cases that stress relocate / approach / board-target / noise
  full    quick + perp seeds 0-9 + offset/aruco-ish targets + mild angled
  stress  full + wider seeds + stronger angled

Examples:
  python scripts/eval_suite.py --preset quick
  python scripts/eval_suite.py --preset bugs --min-success 0.7
  python scripts/eval_suite.py --preset full --min-success 0.8
  python scripts/eval_suite.py --cases smoke_default,perp_0,perp_3 --skip-pytest
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as stats
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "run_drill_task.py"
OUT = ROOT / "outputs"
DEFAULT_METRICS = OUT / "eval_suite.csv"
DEFAULT_REPORT = OUT / "eval_suite_report.txt"
DEFAULT_JSON = OUT / "eval_suite_results.json"


def _case(name, *, random=None, fixed_board=False, tilt=0.0, azim=0.0,
          allow_angled=False, board_target="marker", no_noise=False,
          no_place=True, extra=None, note=""):
    return {
        "name": name,
        "random": random,
        "fixed_board": fixed_board,
        "tilt": float(tilt),
        "azim": float(azim),
        "allow_angled": bool(allow_angled),
        "board_target": board_target,
        "no_noise": bool(no_noise),
        "no_place": bool(no_place),
        "extra": list(extra or []),
        "note": note,
    }


# Named scenarios. Keep names stable — reports/CI can key off them.
CASES = {
    "smoke_default": _case(
        "smoke_default",
        note="default drill/board spawn (recent approach-far tracking fail)"),
    "smoke_seed0": _case("smoke_seed0", random=0, note="random seed 0"),
    "smoke_seed3": _case("smoke_seed3", random=3, note="random seed 3"),
    "smoke_seed7": _case("smoke_seed7", random=7, note="random seed 7"),
    "fixed_board_0": _case(
        "fixed_board_0", random=0, fixed_board=True,
        note="jitter drill only; board already fixed"),
    "fixed_board_3": _case(
        "fixed_board_3", random=3, fixed_board=True,
        note="jitter drill only; board already fixed"),
    "target_offset": _case(
        "target_offset", random=1, board_target="offset",
        note="pierce from calib hole_yz, not marker blob"),
    "target_marker_nonoise": _case(
        "target_marker_nonoise", random=2, no_noise=True,
        note="ideal depth (isolates geometry vs D435i noise)"),
    "angled_mild": _case(
        "angled_mild", random=4, tilt=15.0, azim=10.0, allow_angled=True,
        note="mild angled hole; needs unlock or --allow-angled"),
    "angled_strong": _case(
        "angled_strong", random=5, tilt=25.0, azim=20.0, allow_angled=True,
        note="near sim angle gate; stress bit/board clearance"),
    **{
        f"perp_{i}": _case(f"perp_{i}", random=i, note=f"perp baseline seed {i}")
        for i in range(10)
    },
    **{
        f"stress_{i}": _case(f"stress_{i}", random=i, note=f"wider stress seed {i}")
        for i in range(10, 20)
    },
}

PRESETS = {
    "quick": [
        "smoke_default", "smoke_seed0", "smoke_seed3", "fixed_board_0",
    ],
    "bugs": [
        "smoke_default", "smoke_seed0", "smoke_seed3", "smoke_seed7",
        "fixed_board_0", "fixed_board_3",
        "target_offset", "target_marker_nonoise",
        "perp_1", "perp_3", "perp_6",
        "angled_mild",
    ],
    "full": [
        "smoke_default", "smoke_seed0", "smoke_seed3",
        "fixed_board_0", "fixed_board_3",
        "target_offset", "target_marker_nonoise",
        *[f"perp_{i}" for i in range(10)],
        "angled_mild",
    ],
    "stress": [
        "smoke_default",
        *[f"perp_{i}" for i in range(10)],
        *[f"stress_{i}" for i in range(10, 20)],
        "fixed_board_0", "fixed_board_3",
        "target_offset", "target_marker_nonoise",
        "angled_mild", "angled_strong",
    ],
}


def parse_case_list(text: str):
    names = [p.strip() for p in text.split(",") if p.strip()]
    unknown = [n for n in names if n not in CASES]
    if unknown:
        raise SystemExit(f"unknown cases: {unknown}\nknown: {', '.join(sorted(CASES))}")
    return names


def run_pytest(python: str, skip_sim: bool) -> dict:
    cmd = [python, "-m", "pytest", "-q", "--tb=line"]
    if skip_sim:
        # Avoid MuJoCo-heavy modules when DLL/GL is broken.
        ignores = [
            "tests/test_board_spawn.py", "tests/test_board_target.py",
            "tests/test_lidar.py", "tests/test_phase2_no_scene_gt.py",
            "tests/test_phase3_sensors.py", "tests/test_phase4_arm_driver.py",
            "tests/test_phase5_drill_physics.py", "tests/test_grasp_board_clear.py",
            "tests/test_base_camera.py",
        ]
        for p in ignores:
            cmd.extend(["--ignore", p])
    print(f"\n=== pytest ===\n{' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    return {
        "name": "pytest",
        "ok": r.returncode == 0,
        "returncode": r.returncode,
        "fail_reason": "" if r.returncode == 0 else f"pytest exit {r.returncode}",
        "note": "unit/integration tests",
    }


_FAIL_RE = re.compile(r"FAILED:\s*(.+)$", re.M)


def _last_metrics_row(path: Path) -> dict | None:
    if not path.is_file():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    return rows[-1] if rows else None


def _metrics_nrows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as fh:
        return max(0, sum(1 for _ in fh) - 1)  # minus header


def run_case(case: dict, *, python: str, metrics: Path, logs_dir: Path) -> dict:
    name = case["name"]
    log_path = logs_dir / f"{name}.jsonl"
    cmd = [
        python, str(TASK),
        "--eval", "--metrics", str(metrics),
        "--run-log", str(log_path),
        "--board-target", case["board_target"],
        "--tilt", str(case["tilt"]),
        "--azim", str(case["azim"]),
    ]
    if case["random"] is not None:
        cmd.extend(["--random", str(case["random"])])
    if case["fixed_board"]:
        cmd.append("--fixed-board")
    if case["no_place"]:
        cmd.append("--no-place")
    if case["no_noise"]:
        cmd.append("--no-noise")
    if case["allow_angled"] or abs(case["tilt"]) > 1e-9 or abs(case["azim"]) > 1e-9:
        cmd.append("--allow-angled")
    cmd.extend(case["extra"])

    print(f"\n=== {name} ===", flush=True)
    if case["note"]:
        print(f"# {case['note']}", flush=True)
    print(" ".join(cmd), flush=True)
    n_before = _metrics_nrows(metrics)
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    # Keep console readable: print tail of stdout (phase log lives there).
    out = (r.stdout or "") + (r.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-12:])
    if tail:
        print(tail, flush=True)

    row = _last_metrics_row(metrics) if _metrics_nrows(metrics) > n_before else None
    fail_m = _FAIL_RE.search(out)
    if row is not None:
        success = row.get("success") == "1"
        fail_reason = (row.get("fail_reason") or "").strip()
    else:
        success = r.returncode == 0 and fail_m is None
        fail_reason = (fail_m.group(1).strip() if fail_m else
                       (f"exit {r.returncode}" if r.returncode else "no metrics row"))

    if not success and not fail_reason:
        fail_reason = f"exit {r.returncode}" if r.returncode else "unknown failure"

    result = {
        "name": name,
        "ok": bool(success),
        "returncode": r.returncode,
        "fail_reason": fail_reason,
        "note": case.get("note", ""),
        "random": case["random"],
        "tilt": case["tilt"],
        "azim": case["azim"],
        "board_target": case["board_target"],
        "run_log": str(log_path.as_posix()),
    }
    if row:
        for k in ("entry_err_vs_gt_mm", "depth_mm", "approach_lat_mm",
                  "board_hole_err_mm", "sim_time_s", "bit_angle_err_deg"):
            if row.get(k) not in (None, ""):
                result[k] = row[k]
    return result


def _bucket_reason(reason: str) -> str:
    r = (reason or "").lower()
    if not r:
        return "(ok)"
    for key in (
        "tracking error", "joint margin", "workspace", "aim clearance",
        "hand vs board", "trigger", "dry cut", "cut spin", "board contact",
        "angled drilling locked", "perception", "grasp", "lift",
        "hole", "board", "transit", "held clearance",
    ):
        if key in r:
            return key
    # First clause before comma / parenthesis.
    return reason.split("(")[0].strip()[:60] or "other"


def write_report(results: list, path: Path, metrics: Path) -> str:
    task_rows = [r for r in results if r["name"] != "pytest"]
    ok = [r for r in task_rows if r["ok"]]
    fail = [r for r in task_rows if not r["ok"]]
    lines = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines.append(f"eval_suite report  {ts}")
    lines.append(f"metrics: {metrics}")
    lines.append("")
    for r in results:
        if r["name"] == "pytest":
            lines.append(f"pytest: {'PASS' if r['ok'] else 'FAIL'}  {r.get('fail_reason', '')}")
    n = len(task_rows)
    rate = len(ok) / max(n, 1)
    lines.append(f"task cases: {len(ok)}/{n} success ({100 * rate:.0f}%)")
    lines.append("")

    if fail:
        lines.append("failures:")
        for r in fail:
            lines.append(f"  [{r['name']}] {r['fail_reason']}")
        lines.append("")
        buckets = Counter(_bucket_reason(r["fail_reason"]) for r in fail)
        lines.append("fail_reason buckets:")
        for k, c in buckets.most_common():
            lines.append(f"  {c:3d}  {k}")
        lines.append("")

    def mean_sd(key, unit=""):
        xs = []
        for r in ok:
            v = r.get(key)
            if v in (None, ""):
                continue
            try:
                xs.append(float(v))
            except ValueError:
                pass
        if not xs:
            return
        sd = stats.pstdev(xs) if len(xs) > 1 else 0.0
        lines.append(f"  {key:22s} {stats.mean(xs):7.2f} ± {sd:5.2f} {unit}")

    if ok:
        lines.append("successful runs (mean ± std):")
        mean_sd("entry_err_vs_gt_mm", "mm")
        mean_sd("approach_lat_mm", "mm")
        mean_sd("board_hole_err_mm", "mm")
        mean_sd("depth_mm", "mm")
        mean_sd("bit_angle_err_deg", "deg")
        mean_sd("sim_time_s", "s")
        lines.append("")

    lines.append("per-case:")
    for r in task_rows:
        status = "OK" if r["ok"] else "FAIL"
        extra = r["fail_reason"] if not r["ok"] else (
            f"entry={r.get('entry_err_vs_gt_mm', '-')} mm")
        lines.append(f"  {status:4s}  {r['name']:24s}  {extra}")

    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def main():
    ap = argparse.ArgumentParser(
        description="Full sim eval suite: pytest + multi-scenario run_drill_task")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="bugs",
                    help="case set (default: bugs)")
    ap.add_argument("--cases", default=None,
                    help="comma-separated case names (overrides --preset)")
    ap.add_argument("--list", action="store_true", help="list cases/presets and exit")
    ap.add_argument("--metrics", default=str(DEFAULT_METRICS))
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--json-out", default=str(DEFAULT_JSON))
    ap.add_argument("--min-success", type=float, default=None,
                    help="exit 1 if task success rate below this (0-1)")
    ap.add_argument("--skip-pytest", action="store_true")
    ap.add_argument("--pytest-only", action="store_true")
    ap.add_argument("--pytest-skip-sim", action="store_true",
                    help="ignore MuJoCo-heavy tests if DLL/GL is broken")
    ap.add_argument("--append", action="store_true",
                    help="keep existing metrics CSV instead of starting fresh")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="extra args forwarded to every task run (after --)")
    args = ap.parse_args()

    if args.list:
        print("presets:")
        for k, v in PRESETS.items():
            print(f"  {k:8s}  {len(v)} cases: {', '.join(v)}")
        print("\ncases:")
        for k, c in sorted(CASES.items()):
            print(f"  {k:24s}  {c['note']}")
        return

    names = parse_case_list(args.cases) if args.cases else list(PRESETS[args.preset])
    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]
    for n in names:
        CASES[n]["extra"] = list(CASES[n]["extra"]) + extra

    metrics = Path(args.metrics)
    report_path = Path(args.report)
    json_path = Path(args.json_out)
    logs_dir = OUT / "eval_suite_runs"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if not args.append and metrics.exists():
        metrics.unlink()

    results = []
    if not args.skip_pytest:
        results.append(run_pytest(args.python, args.pytest_skip_sim))
        if args.pytest_only:
            text = write_report(results, report_path, metrics)
            print("\n" + text)
            json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            sys.exit(0 if results[0]["ok"] else 1)

    for name in names:
        results.append(run_case(CASES[name], python=args.python,
                                metrics=metrics, logs_dir=logs_dir))

    text = write_report(results, report_path, metrics)
    payload = {
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "preset": None if args.cases else args.preset,
        "cases": names,
        "metrics": str(metrics.as_posix()),
        "report": str(report_path.as_posix()),
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\n" + text)
    print(f"wrote {report_path}")
    print(f"wrote {json_path}")
    print(f"metrics {metrics}")

    task_rows = [r for r in results if r["name"] != "pytest"]
    rate = sum(1 for r in task_rows if r["ok"]) / max(len(task_rows), 1)
    pytest_fail = any(r["name"] == "pytest" and not r["ok"] for r in results)
    if pytest_fail:
        print("FAIL: pytest did not pass")
        sys.exit(1)
    if args.min_success is not None and rate + 1e-9 < args.min_success:
        print(f"FAIL: success rate {rate:.0%} < --min-success {args.min_success:.0%}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
