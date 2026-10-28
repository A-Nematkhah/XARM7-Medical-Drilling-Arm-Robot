import csv
import statistics as st
from pathlib import Path

path = Path(__file__).with_name("metrics.csv")
rows = list(csv.DictReader(path.open(encoding="utf-8")))
needed = ("success", "seed", "entry_err_vs_gt_mm", "depth_mm", "depth_gt_mm")
missing = [k for k in needed if rows and k not in rows[0]]
if missing:
    print(f"WARNING: metrics.csv missing {missing} — archive/rotate the file and re-run.")
ok = [r for r in rows if r.get("success") == "1"]
fail = [r for r in rows if r.get("success") != "1"]
print(f"total rows: {len(rows)}")
print(f"success: {len(ok)}/{len(rows)} ({100 * len(ok) / max(len(rows), 1):.0f}%)")
if fail:
    print("failures:")
    for r in fail:
        print(f"  seed={r['seed']}: {r['fail_reason']}")


def report(label, key, unit=""):
    xs = [float(r[key]) for r in ok if r.get(key) not in ("", None)]
    if not xs:
        return
    mu = st.mean(xs)
    sd = st.pstdev(xs) if len(xs) > 1 else 0.0
    print(f"{label:28s} {mu:7.2f} ± {sd:5.2f} {unit}   (n={len(xs)})")


print()
print("=== successful runs (mean ± std) ===")
report("grip_err", "grip_err_mm", "mm")
report("drill_yaw_err", "drill_yaw_err_deg", "deg")
report("board_hole_err", "board_hole_err_mm", "mm")
report("board_normal_err", "board_normal_err_deg", "deg")
report("approach_lat", "approach_lat_mm", "mm")
report("entry_vs_GT", "entry_err_vs_gt_mm", "mm")
report("entry_vs_cmd", "entry_err_vs_cmd_mm", "mm")
report("entry_vs_marker", "entry_err_vs_marker_mm", "mm")
report("entry_Y", "entry_err_y_mm", "mm")
report("entry_Z", "entry_err_z_mm", "mm")
report("bit_angle_err", "bit_angle_err_deg", "deg")
report("depth (est)", "depth_mm", "mm")
report("depth_err (est)", "depth_err_mm", "mm")
report("depth (GT)", "depth_gt_mm", "mm")
print()
print("per-seed:")
for r in rows:
    e = r.get("entry_err_vs_gt_mm") or "-"
    print(f"  seed {str(r['seed']):>2}: success={r['success']}  entry_GT={e} mm")
